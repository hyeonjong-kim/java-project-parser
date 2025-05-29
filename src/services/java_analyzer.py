import os
import sys
from typing import List, Tuple, Optional, Type
import uuid # Added for unique IDs
import json # Added for JSON operations
import time # Added for polling sleep
import openai # Added for OpenAI Batch API
from tqdm import tqdm
from py2neo import Graph, Node, Relationship
from py2neo.ogm import GraphObject, Property
from tree_sitter import Node as TreeSitterNode # Renamed to avoid conflict
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from collections import defaultdict # Added for defaultdict


# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.java_models import (
    JavaProject,
    JavaPackage,
    JavaClass, JavaMethod, JavaInterface, JavaEnum,
    GraphObject # Ensure GraphObject is imported if needed for isinstance checks, though type() is used later
)
from src.utils.java_parser import JavaParser, walk, find_node_by_type, extract_text, parse_package_declaration, find_node_by_type_in_children
from src.services.embedding_service import EmbeddingService

class CodeExplanation(BaseModel):
    description: str = Field(..., description="자바 코드의 상세 설명 (한국어)")
    summary:     str = Field(..., description="1-2줄 요약 (한국어)")

class JavaAnalyzer:
    """Service for analyzing Java source code and building graph database"""
    
    NODE_TYPE_MAPPING = {
        JavaProject: {"label": "JavaProject", "pk": "name"},
        JavaPackage: {"label": "JavaPackage", "pk": "name"},
        JavaClass: {"label": "JavaClass", "pk": "name"},
        JavaInterface: {"label": "JavaInterface", "pk": "name"},
        JavaEnum: {"label": "JavaEnum", "pk": "name"},
        JavaMethod: {"label": "JavaMethod", "pk": "signature"},
    }

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, openai_api_key: str, 
                 upstage_api_key: Optional[str] = None, postgres_url: Optional[str] = None):
        self.graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.parser = JavaParser()
        self.batch_size = 2000  # Reduced initial batch size for better memory management
        self.pending_nodes = []
        self.pending_relationships = []
        self.openai_api_key = openai_api_key # Store for client initialization
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=openai_api_key, max_tokens=10000)
        self.llm_parser = PydanticOutputParser(pydantic_object=CodeExplanation)
        self.node_cache = {} # In-memory cache for nodes
        self.processed_files_count = 0  # Track processed files for adaptive batching
        self.project = None  # Store the current project instance
        self.project_root_path = None  # Store the project root directory path
        self.internal_packages = set()  # Set of internal package names discovered from directory structure
        
        # Initialize embedding service if credentials are provided
        self.embedding_service = None
        if upstage_api_key and postgres_url:
            try:
                self.embedding_service = EmbeddingService(upstage_api_key, postgres_url)
                print("Embedding service initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize embedding service: {e}")
                print("Continuing without embedding functionality")
        else:
            print("Embedding service not initialized (missing upstage_api_key or postgres_url)")
    
    def _get_explanation_code(self, vertex_info: dict) -> Tuple[str, str]:
        """Get explanation of the code based on vertex_info dictionary.
           vertex_info is expected to have a 'body' key with the code string.
        """
        body = vertex_info.get("body")
        node_name = vertex_info.get("name", "Unnamed Node") # For logging/error messages
        node_labels = vertex_info.get("labels", [])

        additional_context = ""
        is_package_type = "JavaPackage" in node_labels
        is_project_type = "JavaProject" in node_labels

        # Escape curly braces in user-provided content to prevent Langchain from treating them as variables
        escaped_body = body.replace("{", "{{").replace("}", "}}") if body and isinstance(body, str) else None

        if "JavaClass" in node_labels:
            raw_class_context = self._build_incoming_context_for_class(vertex_info)
            additional_context = raw_class_context.replace("{", "{{").replace("}", "}}")
        elif is_package_type:
            raw_package_context = self._build_incoming_context_for_package(vertex_info)
            additional_context = raw_package_context.replace("{", "{{").replace("}", "}}")
        elif is_project_type:
            raw_project_context = self._build_incoming_context_for_project(vertex_info)
            additional_context = raw_project_context.replace("{", "{{").replace("}", "}}")

        # Prepare variables for the prompt
        prompt_variables = {
            "format_instructions": self.llm_parser.get_format_instructions()
        }

        user_prompt_template_parts = []

        if not (is_package_type or is_project_type) and escaped_body and escaped_body.strip():
            user_prompt_template_parts.append("### Java 코드\n{code}\n\n")
            prompt_variables["code"] = escaped_body # Use escaped_body for {code}
        elif (is_package_type or is_project_type) and not (escaped_body and escaped_body.strip()): # Package/Project type, body is not expected to be primary content
            prompt_variables["code"] = "" # Provide empty string for {code} if not available but template expects it
        elif not (escaped_body and escaped_body.strip()) and not (additional_context and additional_context.strip()):
             return (
                    f"{node_name}에 대해 LLM에 전달할 유효한 내용(코드 또는 컨텍스트)이 없습니다.",
                    "설명 생성 불가 (LLM 입력 내용 없음)"
                )
        else: # Non-package/project type without body, but has context. Or other unhandled cases.
             prompt_variables["code"] = "" # Default for {code}

        if additional_context and additional_context.strip():
            user_prompt_template_parts.append("\n### 추가 컨텍스트\n{additional_context}\n\n")
            prompt_variables["additional_context"] = additional_context # Use escaped additional_context
        else:
            prompt_variables["additional_context"] = "" # Ensure it's always defined for the template

        if not user_prompt_template_parts: # If no code and no context was added
            return (
                f"{node_name}에 대해 LLM에 전달할 유효한 내용(코드 또는 컨텍스트)이 없습니다.",
                "설명 생성 불가 (LLM 입력 내용 없음)"
            )

        user_prompt_template_parts.append("### 출력 형식 지침\n{format_instructions}")
        user_prompt_template = "".join(user_prompt_template_parts)

        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system",
                    "당신은 숙련된 자바 개발자입니다. "
                    "현재 bottom-up 방식으로 코드 분석을 진행 중이므로, 이미 분석된 하위 구성 요소들의 정보를 최대한 활용하세요. "
                    "예를 들어, 클래스 분석 시에는 이미 분석된 메서드들의 정보를, "
                    "패키지 분석 시에는 이미 분석된 클래스들의 정보를, "
                    "프로젝트 분석 시에는 이미 분석된 패키지들의 정보를 종합하여 설명하세요. "
                    "사용자가 제공한 내용을 읽고, 어떤 역할을 하는지 친절히 설명하고, "
                    "마지막에 한-두 줄로 핵심을 요약하세요. "
                    "주어진 Java 코드, 추가 컨텍스트, 하위 구성 요소 정보들을 모두 고려하여 설명해주세요. "
                    "반드시 제공된 JSON 스키마에 맞춰 응답하세요."),
                    ("user", user_prompt_template)
                ]
            )

            chain: Runnable = prompt | self.llm | self.llm_parser
            # Pass all necessary variables for the formatted prompt to invoke
            result: CodeExplanation = chain.invoke(prompt_variables)
            
            if not result or not result.description or not result.summary:
                # print(f"LLM returned no valid explanation for {node_name}.")
                return (
                    f"{node_name} 코드 분석 중 오류가 발생했거나 유효한 설명을 생성하지 못했습니다.",
                    "설명 생성 오류 (LLM 반환 값 문제)"
                )
            
            # print(f"Received LLM explanation for {node_name}.")
            return result.description, result.summary
            
        except Exception as e:
            print(f"Error analyzing code for {node_name} with LLM: {str(e)}")
            return (
                f"{node_name}에 대한 LLM 분석 중 예외가 발생하여 설명을 생성하지 못했습니다: {str(e)}",
                "LLM 분석 예외 발생"
            )
    
    def _build_incoming_context_for_class(self, vertex_info: dict) -> str:
        """Builds a string context of in-coming relationships and connected nodes for a given vertex."""
        target_name = vertex_info.get("name")
        target_labels = vertex_info.get("labels")

        if not target_name or not target_labels:
            return "대상 정점의 이름이나 레이블 정보가 없어 In-coming 컨텍스트를 생성할 수 없습니다."

        # Assuming the first label in the list is the primary label for querying
        # This might need adjustment if a node can have multiple primary-like labels from our list
        target_label = target_labels[0] if isinstance(target_labels, (list, tuple)) and target_labels else None
        if not target_label:
             return f"대상 정점 '{target_name}'의 유효한 주 레이블이 없어 In-coming 컨텍스트를 생성할 수 없습니다."

        context_lines = []
        context_lines.append(f"타겟 정점 '{target_name}' ({target_label}) 의 In-coming 컨텍스트:")

        tx = self.graph.begin()
        try:
            # Query for in-coming relationships
            # We need to match the target node using its primary label and name property.
            # Ensure target_name is properly escaped or handled if it can contain special Cypher characters (though names usually don't)
            query = (
                f"MATCH (s)-[r]->(t:{target_label} {{name: $target_name}}) "
                "RETURN s.name AS source_name, labels(s) AS source_labels, type(r) AS relation_type, "
                "s.summary AS source_summary, s.description AS source_description"
            )
            
            result = tx.run(query, target_name=target_name)
            found_incoming = False
            for record in result:
                found_incoming = True
                source_name = record["source_name"]
                source_labels_list = record["source_labels"]
                relation_type = record["relation_type"]
                source_summary = record.get("source_summary", "요약 정보 없음")
                source_description = record.get("source_description", "설명 정보 없음")
                
                # Take the first label for simplicity, or join them
                display_source_label = source_labels_list[0] if source_labels_list else "UnknownLabel"

                context_lines.append(
                    f"  - '{source_name}' ({display_source_label}) --[{relation_type}]--> '{target_name}'"
                )
                context_lines.append(f"    요약: {source_summary if source_summary else 'N/A'}")
                context_lines.append(f"    설명: {source_description if source_description else 'N/A'}")
            
            if not found_incoming:
                context_lines.append("  (In-coming 관계가 없습니다.)")

            # For bottom-up analysis, also include information about methods contained in this class
            if target_label == "JavaClass":
                methods_query = (
                    "MATCH (m:JavaMethod)-[:METHOD]->(c:JavaClass {name: $target_name}) "
                    "RETURN m.name AS method_name, m.signature AS method_signature, "
                    "m.summary AS method_summary, m.description AS method_description "
                    "ORDER BY m.name"
                )
                
                methods_result = tx.run(methods_query, target_name=target_name)
                methods_found = False
                method_lines = []
                
                for method_record in methods_result:
                    if not methods_found:
                        method_lines.append("\n포함된 메서드들 (이미 분석됨):")
                        methods_found = True
                    
                    method_name = method_record["method_name"]
                    method_signature = method_record["method_signature"]
                    method_summary = method_record.get("method_summary", "요약 없음")
                    method_description = method_record.get("method_description", "설명 없음")
                    
                    method_lines.append(f"  메서드: {method_signature}")
                    if method_summary and method_summary != "요약 없음":
                        method_lines.append(f"    요약: {method_summary}")
                    if method_description and method_description != "설명 없음":
                        method_lines.append(f"    설명: {method_description[:100]}{'...' if len(method_description) > 100 else ''}")
                
                context_lines.extend(method_lines)
            
            # tx.commit() # Read-only, commit not strictly needed
            return "\n".join(context_lines)

        except Exception as e:
            if tx and not tx.closed: # Check if closed before rollback
                tx.rollback() # Rollback on error
            return f"'{target_name}' ({target_label}) 에 대한 In-coming 컨텍스트 생성 중 오류 발생: {str(e)}"
        finally:
            # Ensure transaction is closed if it was opened and not by commit/rollback in try/except
            if tx and not tx.closed:
                try:
                    tx.rollback() # Or tx.finish() - safer to rollback if uncertain
                    # print(f"Transaction for '{target_name}' (class context) rolled back in finally.")
                except Exception as e_tx_final:
                    print(f"Error in finally block rolling back tx for '{target_name}' (class context): {e_tx_final}")

    def _build_incoming_context_for_package(self, vertex_info: dict) -> str:
        """Builds a string context of in-coming relationships and connected nodes for a given vertex."""
        target_name = vertex_info.get("name")
        target_labels = vertex_info.get("labels")

        if not target_name or not target_labels:
            return "대상 정점의 이름이나 레이블 정보가 없어 In-coming 컨텍스트를 생성할 수 없습니다."

        # Assuming the first label in the list is the primary label for querying
        # This might need adjustment if a node can have multiple primary-like labels from our list
        target_label = target_labels[0] if isinstance(target_labels, (list, tuple)) and target_labels else None
        if not target_label:
             return f"대상 정점 '{target_name}'의 유효한 주 레이블이 없어 In-coming 컨텍스트를 생성할 수 없습니다."

        context_lines = []
        context_lines.append(f"타겟 정점 '{target_name}' ({target_label}) 의 In-coming 컨텍스트:")

        tx = self.graph.begin()
        try:
            query = (
                f"MATCH (s)-[r]->(t:{target_label} {{name: $target_name}}) "
                "RETURN s.name AS source_name, labels(s) AS source_labels, type(r) AS relation_type, "
                "s.summary AS source_summary, s.description AS source_description"
            )
            
            result = tx.run(query, target_name=target_name)
            found_incoming = False
            for record in result:
                found_incoming = True
                source_name = record["source_name"]
                source_labels_list = record["source_labels"]
                relation_type = record["relation_type"]
                source_summary = record.get("source_summary", "요약 정보 없음")
                
                display_source_label = source_labels_list[0] if source_labels_list else "UnknownLabel"

                context_lines.append(
                    f"  - '{source_name}' ({display_source_label}) --[{relation_type}]--> '{target_name}'"
                )
                context_lines.append(f"    요약: {source_summary if source_summary else 'N/A'}")
            
            if not found_incoming:
                context_lines.append("  (In-coming 관계가 없습니다.)")

            # For bottom-up analysis, include information about classes/interfaces/enums contained in this package
            contained_elements_query = (
                "MATCH (element)-[rel]->(p:JavaPackage {name: $target_name}) "
                "WHERE type(rel) IN ['CLASS', 'INTERFACE', 'ENUM'] "
                "RETURN labels(element) AS element_labels, element.name AS element_name, "
                "element.summary AS element_summary, element.description AS element_description, "
                "type(rel) AS relation_type "
                "ORDER BY relation_type, element.name"
            )
            
            elements_result = tx.run(contained_elements_query, target_name=target_name)
            elements_found = False
            element_lines = []
            
            for element_record in elements_result:
                if not elements_found:
                    element_lines.append("\n포함된 요소들 (이미 분석됨):")
                    elements_found = True
                
                element_name = element_record["element_name"]
                element_labels = element_record["element_labels"]
                relation_type = element_record["relation_type"]
                element_summary = element_record.get("element_summary", "요약 없음")
                element_description = element_record.get("element_description", "설명 없음")
                
                element_type = element_labels[0] if element_labels else "UnknownType"
                element_lines.append(f"  {relation_type}: {element_name} ({element_type})")
                
                if element_summary and element_summary != "요약 없음":
                    element_lines.append(f"    요약: {element_summary}")
                if element_description and element_description != "설명 없음":
                    element_lines.append(f"    설명: {element_description[:100]}{'...' if len(element_description) > 100 else ''}")
            
            context_lines.extend(element_lines)
            
            # tx.commit() # Read-only
            return "\n".join(context_lines)
            
        except Exception as e:
            if tx and not tx.closed: # Check if closed before rollback
                tx.rollback() # Rollback on error
            return f"'{target_name}' ({target_label}) 에 대한 In-coming 컨텍스트 생성 중 오류 발생: {str(e)}"
        finally:
            # Ensure transaction is closed if it was opened and not by commit/rollback in try/except
            if tx and not tx.closed:
                try:
                    tx.rollback() # Or tx.finish() - safer to rollback if uncertain
                    # print(f"Transaction for '{target_name}' (package context) rolled back in finally.")
                except Exception as e_tx_final:
                    print(f"Error in finally block rolling back tx for '{target_name}' (package context): {e_tx_final}")

    def _build_incoming_context_for_project(self, vertex_info: dict) -> str:
        """Builds a string context of in-coming relationships and connected nodes for a given project vertex."""
        target_name = vertex_info.get("name")
        target_labels = vertex_info.get("labels")

        if not target_name or not target_labels:
            return "대상 정점의 이름이나 레이블 정보가 없어 In-coming 컨텍스트를 생성할 수 없습니다."

        target_label = target_labels[0] if isinstance(target_labels, (list, tuple)) and target_labels else None
        if not target_label:
             return f"대상 정점 '{target_name}'의 유효한 주 레이블이 없어 In-coming 컨텍스트를 생성할 수 없습니다."

        context_lines = []
        context_lines.append(f"타겟 정점 '{target_name}' ({target_label}) 의 In-coming 컨텍스트:")

        tx = self.graph.begin()
        try:
            # For projects, we want to see what packages are contained within it
            query = (
                f"MATCH (s)-[r]->(t:{target_label} {{name: $target_name}}) "
                "RETURN s.name AS source_name, labels(s) AS source_labels, type(r) AS relation_type, "
                "s.summary AS source_summary, s.description AS source_description"
            )
            
            result = tx.run(query, target_name=target_name)
            found_incoming = False
            for record in result:
                found_incoming = True
                source_name = record["source_name"]
                source_labels_list = record["source_labels"]
                relation_type = record["relation_type"]
                source_summary = record.get("source_summary", "요약 정보 없음")
                
                display_source_label = source_labels_list[0] if source_labels_list else "UnknownLabel"

                context_lines.append(
                    f"  - '{source_name}' ({display_source_label}) --[{relation_type}]--> '{target_name}'"
                )
                context_lines.append(f"    요약: {source_summary if source_summary else 'N/A'}")
            
            if not found_incoming:
                context_lines.append("  (In-coming 관계가 없습니다.)")

            # For bottom-up analysis, include information about packages connected to this project
            packages_query = (
                "MATCH (pkg:JavaPackage)-[:PACKAGE]->(proj:JavaProject {name: $target_name}) "
                "RETURN pkg.name AS package_name, pkg.is_internal AS is_internal, "
                "pkg.summary AS package_summary, pkg.description AS package_description "
                "ORDER BY pkg.name"
            )
            
            packages_result = tx.run(packages_query, target_name=target_name)
            packages_found = False
            package_lines = []
            
            for package_record in packages_result:
                if not packages_found:
                    package_lines.append("\n포함된 패키지들 (이미 분석됨):")
                    packages_found = True
                
                package_name = package_record["package_name"]
                is_internal = package_record.get("is_internal", False)
                package_summary = package_record.get("package_summary", "요약 없음")
                package_description = package_record.get("package_description", "설명 없음")
                
                pkg_type = "내부" if is_internal else "외부"
                package_lines.append(f"  패키지: {package_name} ({pkg_type})")
                
                if package_summary and package_summary != "요약 없음":
                    package_lines.append(f"    요약: {package_summary}")
                if package_description and package_description != "설명 없음":
                    package_lines.append(f"    설명: {package_description[:100]}{'...' if len(package_description) > 100 else ''}")
            
            context_lines.extend(package_lines)

            # Also provide a summary of the project structure
            if packages_found:
                summary_query = (
                    "MATCH (pkg:JavaPackage)-[:PACKAGE]->(proj:JavaProject {name: $target_name}) "
                    "OPTIONAL MATCH (cls:JavaClass)-[:CLASS]->(pkg) "
                    "OPTIONAL MATCH (iface:JavaInterface)-[:INTERFACE]->(pkg) "
                    "OPTIONAL MATCH (enum:JavaEnum)-[:ENUM]->(pkg) "
                    "OPTIONAL MATCH (method:JavaMethod)-[:METHOD]->(cls) "
                    "RETURN count(DISTINCT pkg) as package_count, "
                    "count(DISTINCT cls) as class_count, "
                    "count(DISTINCT iface) as interface_count, "
                    "count(DISTINCT enum) as enum_count, "
                    "count(DISTINCT method) as method_count"
                )
                
                summary_result = tx.run(summary_query, target_name=target_name)
                summary_record = list(summary_result)[0]
                
                context_lines.append("\n프로젝트 구조 요약:")
                context_lines.append(f"  총 패키지: {summary_record['package_count']}개")
                context_lines.append(f"  총 클래스: {summary_record['class_count']}개")
                context_lines.append(f"  총 인터페이스: {summary_record['interface_count']}개")
                context_lines.append(f"  총 열거형: {summary_record['enum_count']}개")
                context_lines.append(f"  총 메서드: {summary_record['method_count']}개")
            
            # tx.commit() # Read-only
            return "\n".join(context_lines)
            
        except Exception as e:
            if tx and not tx.closed:
                tx.rollback()
            return f"'{target_name}' ({target_label}) 에 대한 In-coming 컨텍스트 생성 중 오류 발생: {str(e)}"
        finally:
            if tx and not tx.closed:
                try:
                    tx.rollback()
                except Exception as e_tx_final:
                    print(f"Error in finally block rolling back tx for '{target_name}' (project context): {e_tx_final}")

    def _flush_batch(self):
        """Flush pending nodes and relationships to Neo4j using UNWIND for OGM objects where possible."""
        # Early return if nothing to process
        if not self.pending_nodes and not self.pending_relationships:
            return

        # Log batch size for monitoring
        total_items = len(self.pending_nodes) + len(self.pending_relationships)
        print(f"Flushing batch: {len(self.pending_nodes)} nodes, {len(self.pending_relationships)} relationships (total: {total_items})")

        tx = self.graph.begin()
        try:
            nodes_to_unwind_by_label = defaultdict(list)
            ogm_objects_for_final_ogm_merge = []
            raw_nodes_to_create = []

            # Phase 0: Categorize all pending nodes for optimal processing
            for item in self.pending_nodes:
                item_type = type(item)
                if item_type in self.NODE_TYPE_MAPPING:
                    node_type_info = self.NODE_TYPE_MAPPING[item_type]
                    label = node_type_info["label"]
                    pk_name = node_type_info["pk"]
                    
                    # Extract properties from OGM object safely
                    props = {}
                    for attr_name, attr_value in item_type.__dict__.items():
                        if isinstance(attr_value, Property):
                            try:
                                props[attr_name] = getattr(item, attr_name)
                            except AttributeError:
                                props[attr_name] = None
                    
                    if pk_name not in props or props[pk_name] is None:
                        print(f"Warning: OGM object {item} of type {item_type.__name__} is missing primary key '{pk_name}' or it's None. Will be handled by standard OGM merge.")
                        ogm_objects_for_final_ogm_merge.append(item)
                        continue

                    nodes_to_unwind_by_label[label].append(props)
                    ogm_objects_for_final_ogm_merge.append(item)
                
                elif isinstance(item, GraphObject):
                    ogm_objects_for_final_ogm_merge.append(item)
                elif isinstance(item, Node):
                    raw_nodes_to_create.append(item)

            # Phase 1: Batch UNWIND operations by label for maximum efficiency
            successful_labels = set()
            for label, props_list in nodes_to_unwind_by_label.items():
                if not props_list:
                    continue
                
                pk_name = None
                for ogm_type_class, type_info_dict in self.NODE_TYPE_MAPPING.items():
                    if type_info_dict["label"] == label:
                        pk_name = type_info_dict["pk"]
                        break
                
                if not pk_name:
                    print(f"Critical Error: Could not find primary key for label {label} during UNWIND. Skipping UNWIND for this type.")
                    continue

                query = f"UNWIND $props_list AS props MERGE (n:{label} {{{pk_name}: props.{pk_name}}}) SET n = props"
                try:
                    tx.run(query, props_list=props_list)
                    successful_labels.add(label)
                except Exception as e:
                    print(f"Error during UNWIND for label {label}: {e}. Will skip OGM merge for affected objects.")
                    # Don't continue with OGM merge if UNWIND failed critically
                    if "ConstraintValidationFailed" in str(e):
                        print(f"Constraint violation for {label}. This suggests schema changes are needed.")

            # Phase 1.5: Batch create raw py2neo.Node objects
            if raw_nodes_to_create:
                try:
                    for node_to_create in raw_nodes_to_create:
                        tx.create(node_to_create)
                except Exception as e:
                    print(f"Error creating raw nodes: {e}")

            # Phase 2: OGM merge for relationships and fallbacks (only for successful labels)
            if not hasattr(tx, '_closed') or not tx._closed:
                # Filter OGM objects to only include those from successful UNWIND operations
                filtered_ogm_objects = []
                for obj in ogm_objects_for_final_ogm_merge:
                    obj_type = type(obj)
                    if obj_type in self.NODE_TYPE_MAPPING:
                        label = self.NODE_TYPE_MAPPING[obj_type]["label"]
                        if label in successful_labels or label not in nodes_to_unwind_by_label:
                            filtered_ogm_objects.append(obj)
                    else:
                        filtered_ogm_objects.append(obj)

                for obj in filtered_ogm_objects:
                    try:
                        tx.merge(obj)
                    except Exception as e:
                        obj_id = obj.__primaryvalue__ if hasattr(obj, '__primaryvalue__') else str(obj)
                        print(f"Error during OGM merge for {type(obj).__name__} (ID/PK: {obj_id}): {e}")

                # Phase 3: Batch create relationships
                if self.pending_relationships:
                    try:
                        for rel in self.pending_relationships:
                            tx.create(rel)
                    except Exception as e:
                        print(f"Error creating relationships: {e}")

            # Commit only if transaction is still valid
            if not hasattr(tx, '_closed') or not tx._closed:
                self.graph.commit(tx)
                print(f"Successfully flushed batch of {total_items} items")
            else:
                print("Transaction was closed due to errors. Cannot commit.")
                
        except Exception as e:
            if tx and (not hasattr(tx, '_closed') or not tx._closed):
                try:
                    self.graph.rollback(tx)
                except:
                    pass  # Transaction might already be closed
            print(f"Transaction failed in _flush_batch: {e}")
            # Don't re-raise to allow processing to continue

        finally:
            # Clear caches and pending items
            self.pending_nodes.clear()
            self.pending_relationships.clear()
            self.node_cache.clear()
    
    def _add_to_batch(self, node=None, relationship=None):
        """Add node or relationship to pending batch with adaptive sizing"""
        if node:
            self.pending_nodes.append(node)
        if relationship:
            self.pending_relationships.append(relationship)
        
        # Adaptive batch sizing based on performance
        current_batch_size = len(self.pending_nodes) + len(self.pending_relationships)
        
        # Use smaller batches for the first few files to avoid memory issues
        if self.processed_files_count < 5:
            flush_threshold = min(self.batch_size // 2, 1000)
        else:
            flush_threshold = self.batch_size
            
        if current_batch_size >= flush_threshold:
            self._flush_batch()
    
    def get_or_create_project(self, project_name: str) -> JavaProject:
        """Get existing project or create new one. Project has empty summary/description initially."""
        cache_key = (JavaProject, project_name)
        if cache_key in self.node_cache:
            return self.node_cache[cache_key]

        project = JavaProject.match(self.graph, project_name).first()
        if not project:
            project = JavaProject(project_name, description="", summary="")
            self._add_to_batch(node=project)
        
        if project:  # Ensure project is not None before caching
            self.node_cache[cache_key] = project
            self.project = project  # Store current project instance
        return project
    
    def get_or_create_package(self, package_name: str) -> JavaPackage:
        """Get existing package or create new one. Packages have empty summary/description initially."""
        cache_key = (JavaPackage, package_name) 
        if cache_key in self.node_cache:
            return self.node_cache[cache_key]

        package = JavaPackage.match(self.graph, package_name).first()
        if not package:
            # Determine if this is an internal package
            is_internal = self._is_internal_package(package_name)
            
            package = JavaPackage(package_name, description="", summary="", is_internal=is_internal)
            self._add_to_batch(node=package)
            
            # Log package type for debugging
            package_type = "internal" if is_internal else "external"
        
        if package: # Ensure package is not None before caching
            self.node_cache[cache_key] = package
        return package
    
    def get_or_create_class(self, class_name: str, body: str = "", description: str = "", summary: str = "") -> JavaClass:
        """Get existing class or create new one"""
        cache_key = (JavaClass, class_name)
        if cache_key in self.node_cache:
            return self.node_cache[cache_key]
            
        cls = JavaClass.match(self.graph, class_name).first()
        if not cls:
            cls = JavaClass(class_name, body, description, summary)
            self._add_to_batch(node=cls)
        
        if cls: # Ensure cls is not None before caching
            self.node_cache[cache_key] = cls
        return cls
    
    def get_or_create_interface(self, interface_name: str, body: str = "", description: str = "", summary: str = "") -> JavaInterface:
        """Get existing interface or create new one"""
        cache_key = (JavaInterface, interface_name)
        if cache_key in self.node_cache:
            return self.node_cache[cache_key]

        interface = JavaInterface.match(self.graph, interface_name).first()
        if not interface:
            interface = JavaInterface(interface_name, body, description, summary)
            self._add_to_batch(node=interface)
        
        if interface: # Ensure interface is not None before caching
            self.node_cache[cache_key] = interface
        return interface
    
    def get_or_create_method(self, method_name: str, signature: str, body: str = "", description: str = "", return_type: str = "", summary: str = "") -> JavaMethod:
        """Get existing method or create new one"""
        cache_key = (JavaMethod, signature) # Use signature for caching instead of method_name
        if cache_key in self.node_cache:
            return self.node_cache[cache_key]

        method = JavaMethod.match(self.graph, signature).first()
        if not method:
            method = JavaMethod(method_name, signature, body, description, summary, return_type)
            self._add_to_batch(node=method)

        if method: # Ensure method is not None before caching
            self.node_cache[cache_key] = method
        return method

    def get_or_create_enum(self, enum_name: str, body: str = "", description: str = "", summary: str = "") -> JavaEnum:
        """Get existing enum or create new one"""
        cache_key = (JavaEnum, enum_name)
        if cache_key in self.node_cache:
            return self.node_cache[cache_key]

        enum = JavaEnum.match(self.graph, enum_name).first()
        if not enum:
            enum = JavaEnum(enum_name, body, description, summary)
            self._add_to_batch(node=enum)

        if enum: # Ensure enum is not None before caching
            self.node_cache[cache_key] = enum
        return enum

    def process_imports(self, root: TreeSitterNode, leaf_package: JavaPackage) -> None:
        """Process import declarations"""
        for imp in (n for n in root.children if n.type == "import_declaration"):
            imp_text = extract_text(imp).strip()
            is_static = " static " in imp_text
            path = imp_text.replace("import", "").replace(";", "").strip()
            
            if is_static:
                # Ensure correct splitting for static imports like "import static java.util.Collections.emptyList;"
                fq_parts = path.split(" ", 1)
                if len(fq_parts) > 1:
                    fq = fq_parts[1]
                else: # Should not happen for valid static imports
                    print(f"Warning: Could not parse FQ from static import: {imp_text}")
                    continue
            elif path.endswith(".*"):
                fq = path[:-2]  # Package import like java.util.*
            else:
                fq = path  # Class import like java.io.IOException
                
            # Determine if this is a package import or class import
            if path.endswith(".*"):
                # This is a package import (e.g., java.util.*)
                package_name = fq
            else:
                # This is likely a class import (e.g., java.io.IOException)
                # We need to extract the package name by removing the last component
                parts = fq.split('.')
                if len(parts) > 1:
                    # The last part is likely the class name, everything before is the package
                    package_name = '.'.join(parts[:-1])
                else:
                    # Single component, treat as package (though unusual)
                    package_name = fq
                
            # Create or get the imported package directly
            imported_package = self.get_or_create_package(package_name)
            if imported_package:
                # Create import relationship between packages
                leaf_package.imported_by.add(imported_package)
                self._add_to_batch(node=leaf_package) # Mark leaf_package as needing update
                self._add_to_batch(node=imported_package) # Mark imported package as needing update

    def find_extended_class(self, node: TreeSitterNode) -> JavaClass:
        """Extract extended class information"""
        extended_class = None
        # tree-sitter node for superclass is "superclass"
        # its child is "type_identifier" which contains the name
        type_identifier_node = find_node_by_type_in_children(node, "type_identifier")
        if type_identifier_node:
            extended_class_name = extract_text(type_identifier_node)
            extended_class = self.get_or_create_class(extended_class_name) # Body will be empty for now if not in project
        return extended_class
    
    def find_implemented_interface(self, node: TreeSitterNode) -> List[JavaInterface]:
        """Extract implemented interfaces information"""
        implemented_interfaces: List[JavaInterface] = []
        for child in walk(node):
            if child.type == "type_list":
                for implemented_interface_type in extract_text(child).split(","):
                    implemented_interface = self.get_or_create_interface(implemented_interface_type)
                    implemented_interfaces.append(implemented_interface)
                break
        return implemented_interfaces

    def _read_source_code(self, file_path: str) -> str:
        """Read source code from file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _process_class_node(self, class_node: TreeSitterNode, leaf_package: JavaPackage) -> None:
        """Process a class declaration node"""
        body = extract_text(class_node)
        name_node = find_node_by_type_in_children(class_node, "identifier")
        if not name_node:
            return
        cls_name = extract_text(name_node)
        
        # Initialize with empty description/summary
        java_class = self.get_or_create_class(class_name=cls_name, body=body, description="", summary="")
        
        java_class.contained_in.add(leaf_package)
        self._add_to_batch(node=java_class) 

        for class_child_node in walk(class_node): 
            self._process_class_child_node(class_child_node, java_class, class_node)
    
    def _process_interface_node(self, interface_node: TreeSitterNode, leaf_package: JavaPackage) -> None:
        """Process a interface declaration node"""
        body = extract_text(interface_node)
        name_node = find_node_by_type_in_children(interface_node, "identifier")
        if not name_node:
            return

        interface_name = extract_text(name_node)
        java_interface = self.get_or_create_interface(interface_name, body=body, description="", summary="")

        java_interface.contained_in.add(leaf_package)
        self._add_to_batch(node=java_interface)

    def _process_enum_node(self, enum_node: TreeSitterNode, leaf_package: JavaPackage) -> None:
        """Process a enum declaration node"""
        body = extract_text(enum_node)
        name_node = find_node_by_type_in_children(enum_node, "identifier")
        if not name_node:
            return

        enum_name = extract_text(name_node)
        java_enum = self.get_or_create_enum(enum_name, body=body, description="", summary="")
        
        java_enum.contained_in.add(leaf_package)
        self._add_to_batch(node=java_enum)
        
        for enum_child_node in walk(enum_node):
            self._process_enum_child_node(enum_child_node, java_enum)

    def _process_class_child_node(self, class_child_node: TreeSitterNode, java_class: JavaClass, parent_class_node: TreeSitterNode) -> None:
        """Process a child node of a class declaration. Added parent_class_node for context if needed."""
        if class_child_node.type == "superclass" and class_child_node.parent == parent_class_node: # Ensure it's a direct child
            extended_class = self.find_extended_class(class_child_node)
            if extended_class: # Ensure extended_class is not None
                java_class.extends.add(extended_class)
                self._add_to_batch(node=java_class) # Mark for update

        elif class_child_node.type == "super_interfaces" and class_child_node.parent == parent_class_node: # Ensure it's a direct child
            implemented_interfaces = self.find_implemented_interface(class_child_node)
            for implemented_interface in implemented_interfaces:
                if implemented_interface: # Ensure not None
                    java_class.implements.add(implemented_interface)
            self._add_to_batch(node=java_class) # Mark for update

        elif class_child_node.type == "method_declaration" and class_child_node.parent == find_node_by_type_in_children(parent_class_node, "class_body"): # More specific parent check
            self._process_method_node(class_child_node, java_class)

    def _process_interface_child_node(self, interface_child_node: TreeSitterNode, java_interface: JavaInterface) -> None:
        # Example: processing constant fields or default methods in interfaces
        if interface_child_node.type == "method_declaration": # default or static methods in interfaces
             self._process_method_node(interface_child_node, java_interface) # Pass java_interface
        # Add more processing for other interface elements if needed (e.g., constant_declaration)

    def _process_enum_child_node(self, enum_child_node: TreeSitterNode, java_enum: JavaEnum) -> None:
        # Enum constants and other enum body elements can be processed here if needed in the future
        pass

    def _process_method_node(self, method_node: TreeSitterNode, owner_class_or_interface: JavaClass | JavaInterface) -> None:
        """Process a method declaration node. Owner can be JavaClass or JavaInterface."""
        body_text_full_method = extract_text(method_node) 
        
        method_body_node_for_llm = find_node_by_type_in_children(method_node, "block")

        name_node = find_node_by_type_in_children(method_node, "identifier")
        if not name_node:
            print(f"Warning: Method name node not found in method_node: {body_text_full_method[:50]}...")
            return

        method_name = extract_text(name_node)
        
        return_type = "void" 
        potential_type_node = None
        for child in method_node.children:
            if child == name_node: 
                break
            if child.type in ["type_identifier", "primitive_type", "void_type", "generic_type", "array_type"]:
                potential_type_node = child
        
        if potential_type_node:
            return_type = extract_text(potential_type_node)
        elif find_node_by_type_in_children(method_node, "void_type"):
             return_type = "void"
        elif isinstance(owner_class_or_interface, JavaClass) and method_name == owner_class_or_interface.name:
            return_type = "constructor" 

        # Generate method signature: method_name(param_type1,param_type2,...)
        parameter_types = []
        formal_params_node = find_node_by_type_in_children(method_node, "formal_parameters")
        if formal_params_node:
            for param_node in formal_params_node.children:
                if param_node.type == "formal_parameter":
                    # Extract parameter type from formal_parameter node
                    param_text = extract_text(param_node).strip()
                    param_parts = param_text.split()
                    if len(param_parts) >= 2:
                        param_type = param_parts[0]  # First part is usually the type
                        parameter_types.append(param_type)
        
        # Create signature: methodName(type1,type2,...)
        signature = f"{method_name}({','.join(parameter_types)})"

        java_method = self.get_or_create_method(method_name, signature, body=body_text_full_method, description="", summary="", return_type=return_type)
        
        if isinstance(owner_class_or_interface, JavaClass):
            owner_class_or_interface.methods.add(java_method)
            java_method.contained_in_class.add(owner_class_or_interface)
        elif isinstance(owner_class_or_interface, JavaInterface):
            # Interfaces can have methods (default, static). 
            owner_class_or_interface.methods.add(java_method)
            java_method.contained_in_interface.add(owner_class_or_interface)
        
        self._add_to_batch(node=owner_class_or_interface) # Mark owner for update
        self._add_to_batch(node=java_method) # Add/update method itself
    

    async def process_java_file(self, file_path: str, flush_immediately: bool = True) -> None:
        """Process a single Java file"""
        try:
            source_code = self._read_source_code(file_path)
            root = self.parser.parse(source_code)
            package_name = parse_package_declaration(root)

            if not package_name:
                print(f"Warning: No package declaration found in {file_path}. Skipping.")
                return

            leaf_package = self.get_or_create_package(package_name)
            if not leaf_package:
                print(f"Warning: Could not create package for {package_name} in {file_path}. Skipping.")
                return
            
            # Ensure leaf_package is indeed a JavaLeafPackage as expected by process_imports and others
            if not isinstance(leaf_package, JavaPackage):
                print(f"Warning: Expected JavaLeafPackage for {package_name} but got {type(leaf_package)}. Attempting to proceed.")

            self.process_imports(root, leaf_package) # leaf_package here must be JavaLeafPackage

            # Iterate through top-level declarations in the compilation unit
            compilation_unit_node = root
            if compilation_unit_node.type == "program": # tree-sitter-java root is 'program'
                for node in compilation_unit_node.children: # Direct children of program node
                    if node.type == "class_declaration":
                        self._process_class_node(node, leaf_package)
                    elif node.type == "interface_declaration":
                        self._process_interface_node(node, leaf_package)
                    elif node.type == "enum_declaration":
                        self._process_enum_node(node, leaf_package)
            
            # Increment processed files counter
            self.processed_files_count += 1
            
            # Only flush immediately if requested (for backwards compatibility)
            if flush_immediately:
                self._flush_batch()

        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            # Still flush on error to avoid accumulating problematic data
            if flush_immediately:
                self._flush_batch()

    def merge_duplicate_packages(self) -> None:
        """This method is no longer needed since we only use JavaPackage."""
        print("No duplicate package merging needed - using only JavaPackage.")
        pass

    def collect_all_analyzed_vertices(self) -> List[dict]:
        """Collects all vertices with specified Java-related labels from the graph."""
        print("Collecting all analyzed vertices (JavaProject, JavaMethod, JavaClass, etc.)...")
        collected_vertices = []
        labels_to_collect = [
            "JavaMethod", "JavaClass", "JavaEnum", "JavaInterface",
            "JavaPackage", "JavaProject"
        ]
        tx = self.graph.begin()
        try:
            for label in labels_to_collect:
                if label == "JavaMethod":
                    # For JavaMethod, use signature as the primary identifier
                    query = f"MATCH (n:{label}) RETURN n.signature AS name, labels(n) AS labels, n.summary AS summary, n.description AS description, n.body AS body"
                else:
                    # Standard query for other labels where the primary identifying field is 'name'
                    query = f"MATCH (n:{label}) RETURN n.name AS name, labels(n) AS labels, n.summary AS summary, n.description AS description, n.body AS body"
                
                result = tx.run(query)
                for record in result:
                    vertex_info = {
                        "name": record["name"],
                        "labels": record["labels"],
                        "summary": record.get("summary"),
                        "description": record.get("description"),
                        "body": record.get("body")
                    }
                    collected_vertices.append(vertex_info)
            print(f"Collected {len(collected_vertices)} vertices.")
            return collected_vertices
        except Exception as e:
            if tx: # tx might not be defined if self.graph.begin() failed
                self.graph.rollback(tx)
            print(f"Error collecting analyzed vertices: {str(e)}")
            return [] # Return empty list on error

    def update_vertex_explanation_in_graph(self, vertex_info: dict, description: str, summary: str) -> bool:
        """Updates the description and summary of a vertex in Neo4j based on its name and label."""
        target_name = vertex_info.get("name")
        target_labels = vertex_info.get("labels")

        if not target_name or not target_labels:
            print(f"Error updating graph: Vertex name or labels missing in vertex_info: {vertex_info.get('name')}")
            return False

        target_label = target_labels[0] if isinstance(target_labels, (list, tuple)) and target_labels else None
        if not target_label:
            print(f"Error updating graph: No valid primary label for vertex: {target_name}")
            return False
        
        # Determine the correct primary key property for each node type
        if target_label == "JavaMethod":
            pk_property_name = "signature"
            # For JavaMethod, target_name contains the signature, not just the method name
        else:
            pk_property_name = "name"

        tx = self.graph.begin()
        try:
            # First, let's check if the node actually exists
            check_query = f"MATCH (n:{target_label} {{{pk_property_name}: $target_name}}) RETURN count(n) as count"
            check_result = tx.run(check_query, target_name=target_name)
            
            # More robust way to get single result
            try:
                records = list(check_result)
                if not records:
                    print(f"DEBUG: No records returned for {target_label} '{target_name}'")
                    tx.rollback()
                    return False
                node_count = records[0]["count"]
            except Exception as e:
                print(f"DEBUG: Error getting check result for {target_label} '{target_name}': {e}")
                tx.rollback()
                return False
            
            if node_count == 0:
                print(f"DEBUG: Node {target_label} '{target_name}' does not exist in database")
                tx.rollback()
                return False
            elif node_count > 1:
                print(f"DEBUG: Multiple nodes found for {target_label} '{target_name}' (count: {node_count})")
            
            # Proceed with update
            query = (
                f"MATCH (n:{target_label} {{{pk_property_name}: $target_name}}) "
                "SET n.description = $description, n.summary = $summary "
                "RETURN count(n) as updated_count"
            )
            result = tx.run(query, target_name=target_name, description=description, summary=summary)
            
            # Get update result more robustly
            try:
                update_records = list(result)
                if update_records and update_records[0]["updated_count"] > 0:
                    tx.commit()
                    return True
                else:
                    print(f"Warning: Node {target_label} '{target_name}' not updated.")
                    tx.rollback()
                    return False
            except Exception as e:
                print(f"Error getting update result for {target_label} '{target_name}': {e}")
                tx.rollback()
                return False
                
        except Exception as e:
            print(f"Error updating explanation for {target_label} '{target_name}' in graph: {str(e)}")
            if tx:
                tx.rollback()
            return False

    def clean_incorrect_package_nodes(self) -> None:
        """Clean up package nodes that are actually class names (e.g., java.io.IOException)"""
        print("Cleaning up incorrectly created package nodes...")
        tx = self.graph.begin()
        try:
            # Common patterns for class names that were incorrectly created as packages
            class_patterns = [
                ".*Exception$",  # Ends with Exception
                ".*Error$",      # Ends with Error
                ".*\\.String$",  # java.lang.String
                ".*\\.Integer$", # java.lang.Integer
                ".*\\.List$",    # java.util.List
                ".*\\.Map$",     # java.util.Map
                ".*\\.Set$",     # java.util.Set
                ".*\\.Object$",  # java.lang.Object
                ".*\\.Class$",   # java.lang.Class
                ".*\\.Thread$",  # java.lang.Thread
                ".*\\.Runnable$", # java.lang.Runnable
                # Add more patterns as needed
            ]
            
            # Find package nodes with names that look like class names
            for pattern in class_patterns:
                query = (
                    "MATCH (p:JavaPackage) "
                    "WHERE p.name =~ $pattern "
                    "RETURN p.name AS name"
                )
                
                result = tx.run(query, pattern=pattern)
                incorrect_packages = [record["name"] for record in result]
                
                # Delete these incorrectly created package nodes
                for package_name in incorrect_packages:
                    print(f"Deleting incorrectly created package: {package_name}")
                    delete_query = (
                        "MATCH (p:JavaPackage {name: $package_name}) "
                        "DETACH DELETE p"
                    )
                    tx.run(delete_query, package_name=package_name)
            
            # Also check for common Java standard library class names
            common_class_names = [
                "java.io.IOException",
                "java.lang.String", 
                "java.lang.Integer",
                "java.lang.Object",
                "java.util.List",
                "java.util.Map",
                "java.util.Set",
                "java.util.ArrayList",
                "java.util.HashMap",
                "java.util.HashSet",
                "org.springframework.web.servlet.NoHandlerFoundException",
                "org.springframework.web.servlet.DispatcherServlet",
                "org.springframework.beans.factory.annotation.Autowired",
                "org.springframework.stereotype.Component",
                "org.springframework.stereotype.Service",
                "org.springframework.stereotype.Repository",
                "org.springframework.stereotype.Controller"
            ]
            
            for class_name in common_class_names:
                check_query = (
                    "MATCH (p:JavaPackage {name: $class_name}) "
                    "RETURN count(p) as count"
                )
                result = tx.run(check_query, class_name=class_name)
                
                # More robust way to get result
                try:
                    records = list(result)
                    if records and records[0]["count"] > 0:
                        print(f"Deleting incorrectly created package: {class_name}")
                        delete_query = (
                            "MATCH (p:JavaPackage {name: $class_name}) "
                            "DETACH DELETE p"
                        )
                        tx.run(delete_query, class_name=class_name)
                except Exception as e:
                    print(f"Error checking/deleting package {class_name}: {e}")
            
            tx.commit()
            print("Cleanup completed successfully.")
            
        except Exception as e:
            if tx:
                tx.rollback()
            print(f"Error during cleanup: {str(e)}")

    def initialize_database_schema(self) -> None:
        """Initialize database with proper constraints and clear existing data"""
        print("Initializing database schema...")
        
        try:
            # Clear all existing data
            print("Clearing existing data...")
            self.graph.run("MATCH (n) DETACH DELETE n")
            
            # Drop existing constraints
            print("Dropping existing constraints...")
            try:
                self.graph.run("DROP CONSTRAINT JavaMethod_name IF EXISTS")
                self.graph.run("DROP CONSTRAINT JavaMethod_signature IF EXISTS")
            except Exception as e:
                print(f"Note: Some constraints may not exist: {e}")
            
            # Create new constraints for the updated schema
            print("Creating new constraints...")
            constraints = [
                "CREATE CONSTRAINT JavaProject_name IF NOT EXISTS FOR (pr:JavaProject) REQUIRE pr.name IS UNIQUE",
                "CREATE CONSTRAINT JavaMethod_signature IF NOT EXISTS FOR (m:JavaMethod) REQUIRE m.signature IS UNIQUE",
                "CREATE CONSTRAINT JavaClass_name IF NOT EXISTS FOR (c:JavaClass) REQUIRE c.name IS UNIQUE",
                "CREATE CONSTRAINT JavaInterface_name IF NOT EXISTS FOR (i:JavaInterface) REQUIRE i.name IS UNIQUE",
                "CREATE CONSTRAINT JavaEnum_name IF NOT EXISTS FOR (e:JavaEnum) REQUIRE e.name IS UNIQUE",
                "CREATE CONSTRAINT JavaPackage_name IF NOT EXISTS FOR (p:JavaPackage) REQUIRE p.name IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    self.graph.run(constraint)
                except Exception as e:
                    print(f"Error creating constraint: {constraint}, Error: {e}")
            
            # Create indexes for efficient querying
            print("Creating indexes...")
            indexes = [
                "CREATE INDEX JavaPackage_is_internal IF NOT EXISTS FOR (p:JavaPackage) ON (p.is_internal)"
            ]
            
            for index in indexes:
                try:
                    self.graph.run(index)
                except Exception as e:
                    print(f"Error creating index: {index}, Error: {e}")
            
            print("Database schema initialized successfully.")
            
        except Exception as e:
            print(f"Error initializing database schema: {e}")
            raise

    def connect_top_level_packages_to_project(self) -> None:
        """Connect only top-level internal packages (those without parent packages and is_internal=true) to the project."""
        if not self.project:
            print("No project found. Skipping package-to-project connection.")
            return
            
        print("Connecting top-level internal packages to project...")
        tx = self.graph.begin()
        try:
            # Find internal packages that are not child packages of any other package
            query = """
            MATCH (p:JavaPackage)
            WHERE p.is_internal = true 
              AND NOT EXISTS {
                MATCH (parent:JavaPackage)-[:PARENT_PACKAGE]->(p)
              }
            RETURN p.name AS package_name
            """
            
            result = tx.run(query)
            top_level_internal_packages = [record["package_name"] for record in result]
            
            # Connect each top-level internal package to the project
            for package_name in top_level_internal_packages:
                connect_query = """
                MATCH (proj:JavaProject {name: $project_name})
                MATCH (pkg:JavaPackage {name: $package_name, is_internal: true})
                MERGE (pkg)-[:PACKAGE]->(proj)
                """
                tx.run(connect_query, project_name=self.project.name, package_name=package_name)
                print(f"Connected internal top-level package '{package_name}' to project '{self.project.name}'")
            
            # Also count external packages that were excluded
            external_query = """
            MATCH (p:JavaPackage)
            WHERE p.is_internal = false 
              AND NOT EXISTS {
                MATCH (parent:JavaPackage)-[:PARENT_PACKAGE]->(p)
              }
            RETURN count(p) AS external_count
            """
            external_result = tx.run(external_query)
            external_count = list(external_result)[0]["external_count"]
            
            tx.commit()
            print(f"Successfully connected {len(top_level_internal_packages)} internal top-level packages to project.")
            print(f"Excluded {external_count} external top-level packages from project connection.")
            
        except Exception as e:
            if tx:
                tx.rollback()
            print(f"Error connecting packages to project: {str(e)}")

    def discover_internal_packages(self, project_root_path: str) -> None:
        """Discover internal packages by scanning the project directory structure for Java files."""
        print(f"Discovering internal packages from project directory: {project_root_path}")
        self.project_root_path = project_root_path
        self.internal_packages.clear()
        
        # Find all Java files and extract their package declarations
        for root, dirs, files in os.walk(project_root_path):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if d not in ['.git', '.gradle', '.idea', 'target', 'build', 'node_modules']]
            
            for file in files:
                if file.endswith('.java'):
                    file_path = os.path.join(root, file)
                    try:
                        # Read and parse the Java file to get package declaration
                        source_code = self._read_source_code(file_path)
                        root_node = self.parser.parse(source_code)
                        package_name = parse_package_declaration(root_node)
                        
                        if package_name:
                            self.internal_packages.add(package_name)
                            
                            # Also add parent packages
                            parts = package_name.split('.')
                            for i in range(1, len(parts)):
                                parent_package = '.'.join(parts[:i])
                                self.internal_packages.add(parent_package)
                                
                    except Exception as e:
                        print(f"Error reading package from {file_path}: {e}")
                        continue
        
        print(f"Discovered {len(self.internal_packages)} internal packages")
        if self.internal_packages:
            print("Sample internal packages:", list(self.internal_packages)[:5])

    def _is_internal_package(self, package_name: str) -> bool:
        """Determine if a package is internal to the project or external library."""
        # If internal packages haven't been discovered yet, assume external for safety
        if not self.internal_packages:
            return False
            
        # Check if the package name exactly matches an internal package
        if package_name in self.internal_packages:
            return True
            
        # Check if the package is a parent of any internal package
        # This handles cases where we import a parent package like "java.util"
        # but the internal packages might contain "com.example.myproject.util"
        for internal_pkg in self.internal_packages:
            if internal_pkg.startswith(package_name + "."):
                return True
                
        # Check common external library prefixes
        external_prefixes = [
            "java.", "javax.", "sun.", "com.sun.",
            "org.apache.", "org.springframework.", "org.junit.",
            "org.slf4j.", "org.hibernate.", "org.mockito.",
            "com.fasterxml.", "com.google.", "com.amazonaws.",
            "io.netty.", "io.swagger.", "org.json.",
            "org.w3c.", "org.xml.", "org.omg."
        ]
        
        for prefix in external_prefixes:
            if package_name.startswith(prefix):
                return False
                
        # If not explicitly internal and doesn't match external patterns,
        # default to external for safety (external libraries are more common in imports)
        return False

    def print_package_statistics(self) -> None:
        """Print statistics about internal vs external packages in the graph."""
        print("\n=== Package Statistics ===")
        tx = self.graph.begin()
        try:
            # Count internal packages
            internal_query = "MATCH (p:JavaPackage {is_internal: true}) RETURN count(p) as count"
            internal_result = tx.run(internal_query)
            internal_count = list(internal_result)[0]["count"]
            
            # Count external packages  
            external_query = "MATCH (p:JavaPackage {is_internal: false}) RETURN count(p) as count"
            external_result = tx.run(external_query)
            external_count = list(external_result)[0]["count"]
            
            # Count packages connected to project
            connected_query = "MATCH (p:JavaPackage)-[:PACKAGE]->(proj:JavaProject) RETURN count(p) as count"
            connected_result = tx.run(connected_query)
            connected_count = list(connected_result)[0]["count"]
            
            print(f"Internal packages: {internal_count}")
            print(f"External packages: {external_count}")
            print(f"Total packages: {internal_count + external_count}")
            print(f"Packages connected to project: {connected_count}")
            
            # Show sample internal packages connected to project
            if connected_count > 0:
                sample_connected_query = """
                MATCH (p:JavaPackage)-[:PACKAGE]->(proj:JavaProject) 
                RETURN p.name as name, p.is_internal as is_internal 
                LIMIT 5
                """
                sample_connected_result = tx.run(sample_connected_query)
                sample_connected = [(record["name"], record["is_internal"]) for record in sample_connected_result]
                print(f"Sample packages connected to project:")
                for name, is_internal in sample_connected:
                    pkg_type = "INTERNAL" if is_internal else "EXTERNAL"
                    print(f"  - {name} ({pkg_type})")
            
            # Show sample internal packages NOT connected to project
            not_connected_internal_query = """
            MATCH (p:JavaPackage {is_internal: true})
            WHERE NOT EXISTS {
                MATCH (p)-[:PACKAGE]->(proj:JavaProject)
            }
            RETURN p.name as name 
            LIMIT 5
            """
            not_connected_internal_result = tx.run(not_connected_internal_query)
            not_connected_internal = [record["name"] for record in not_connected_internal_result]
            
            if not_connected_internal:
                print(f"Sample internal packages NOT connected to project:")
                for name in not_connected_internal:
                    print(f"  - {name} (INTERNAL, child package)")
            
            # Show sample external packages
            if external_count > 0:
                sample_external_query = "MATCH (p:JavaPackage {is_internal: false}) RETURN p.name as name LIMIT 5"
                sample_external_result = tx.run(sample_external_query)
                sample_external = [record["name"] for record in sample_external_result]
                print(f"Sample external packages (not connected to project):")
                for name in sample_external:
                    print(f"  - {name} (EXTERNAL)")
            
            tx.commit()
            
        except Exception as e:
            print(f"Error retrieving package statistics: {e}")
            if tx:
                tx.rollback()
        print("========================\n")

    def process_embeddings_for_vertices(self, vertices: List[dict], project_name: str, batch_size: int = 50) -> Tuple[int, int]:
        """정점들의 임베딩을 생성하고 PostgreSQL에 저장합니다."""
        if not self.embedding_service:
            print("Embedding service not available. Skipping embedding processing.")
            return 0, 0
        
        if not vertices:
            print("No vertices to process for embeddings.")
            return 0, 0
        
        print(f"Processing embeddings for {len(vertices)} vertices...")
        
        success_count = 0
        error_count = 0
        
        # 배치 단위로 처리
        for i in range(0, len(vertices), batch_size):
            batch_vertices = vertices[i:i+batch_size]
            print(f"Processing embedding batch {i//batch_size + 1}/{(len(vertices) + batch_size - 1)//batch_size} ({len(batch_vertices)} items)")
            
            try:
                # 각 정점에 대해 임베딩용 텍스트 준비
                texts = []
                for vertex_info in batch_vertices:
                    text = self.embedding_service.prepare_embedding_text(vertex_info)
                    texts.append(text)
                
                # 배치로 임베딩 생성
                print(f"  Creating embeddings for batch...")
                embeddings = self.embedding_service.create_embeddings_batch(texts)
                
                # PostgreSQL에 저장
                print(f"  Saving embeddings to PostgreSQL...")
                batch_success, batch_error = self.embedding_service.save_embeddings_batch(
                    batch_vertices, embeddings, project_name, batch_size=100
                )
                
                success_count += batch_success
                error_count += batch_error
                
                print(f"  Batch completed: {batch_success} success, {batch_error} errors")
                
            except Exception as e:
                print(f"Error processing embedding batch {i//batch_size + 1}: {e}")
                error_count += len(batch_vertices)
                continue
        
        print(f"\nEmbedding processing completed:")
        print(f"  Total success: {success_count}")
        print(f"  Total errors: {error_count}")
        
        return success_count, error_count

    def update_vertex_with_embedding_info(self, vertex_info: dict) -> dict:
        """정점 정보에 패키지 이름과 기타 임베딩에 필요한 메타데이터를 추가합니다."""
        enhanced_vertex = vertex_info.copy()
        
        node_type = vertex_info.get("labels", ["Unknown"])[0] if vertex_info.get("labels") else "Unknown"
        node_name = vertex_info.get("name", "")
        
        # 패키지 정보 추가
        if node_type in ["JavaClass", "JavaInterface", "JavaEnum"]:
            # 클래스/인터페이스/열거형의 경우 소속 패키지 찾기
            package_name = self._find_package_for_node(node_name, node_type)
            enhanced_vertex["package_name"] = package_name
        elif node_type == "JavaMethod":
            # 메서드의 경우 소속 클래스를 통해 패키지 찾기
            class_name = self._find_class_for_method(node_name)
            if class_name:
                package_name = self._find_package_for_node(class_name, "JavaClass")
                enhanced_vertex["package_name"] = package_name
        elif node_type == "JavaPackage":
            enhanced_vertex["package_name"] = node_name
        
        return enhanced_vertex

    def _find_package_for_node(self, node_name: str, node_type: str) -> Optional[str]:
        """노드가 속한 패키지를 찾습니다."""
        tx = self.graph.begin()
        try:
            if node_type == "JavaClass":
                query = "MATCH (c:JavaClass {name: $node_name})-[:CLASS]->(p:JavaPackage) RETURN p.name as package_name"
            elif node_type == "JavaInterface":
                query = "MATCH (i:JavaInterface {name: $node_name})-[:INTERFACE]->(p:JavaPackage) RETURN p.name as package_name"
            elif node_type == "JavaEnum":
                query = "MATCH (e:JavaEnum {name: $node_name})-[:ENUM]->(p:JavaPackage) RETURN p.name as package_name"
            else:
                return None
            
            result = tx.run(query, node_name=node_name)
            records = list(result)
            if records:
                return records[0]["package_name"]
            return None
            
        except Exception as e:
            print(f"Error finding package for {node_type} {node_name}: {e}")
            return None
        finally:
            if tx and not tx.closed:
                try:
                    tx.rollback()
                except:
                    pass

    def _find_class_for_method(self, method_signature: str) -> Optional[str]:
        """메서드가 속한 클래스를 찾습니다."""
        tx = self.graph.begin()
        try:
            query = "MATCH (m:JavaMethod {signature: $signature})-[:METHOD]->(c:JavaClass) RETURN c.name as class_name"
            result = tx.run(query, signature=method_signature)
            records = list(result)
            if records:
                return records[0]["class_name"]
            return None
            
        except Exception as e:
            print(f"Error finding class for method {method_signature}: {e}")
            return None
        finally:
            if tx and not tx.closed:
                try:
                    tx.rollback()
                except:
                    pass

    def get_embedding_statistics(self, project_name: Optional[str] = None) -> dict:
        """임베딩 통계를 조회합니다."""
        if not self.embedding_service:
            return {"error": "Embedding service not available"}
        
        return self.embedding_service.get_embedding_statistics(project_name)

    def search_similar_code(self, query: str, project_name: Optional[str] = None, 
                           node_types: Optional[List[str]] = None, limit: int = 10,
                           similarity_threshold: float = 0.7) -> List[dict]:
        """유사한 코드를 검색합니다."""
        if not self.embedding_service:
            print("Embedding service not available.")
            return []
        
        return self.embedding_service.search_similar_code(
            query=query,
            project_name=project_name,
            node_types=node_types,
            limit=limit,
            similarity_threshold=similarity_threshold
        )


def find_java_files(directory: str) -> List[str]:
    """Find all Java files in directory"""
    java_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.java'):
                java_files.append(os.path.join(root, file))
    return java_files 