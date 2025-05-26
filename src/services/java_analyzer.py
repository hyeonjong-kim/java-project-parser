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
    JavaPackage, JavaInternalPackage, JavaLeafPackage,
    JavaClass, JavaMethod, JavaField, JavaParameter, JavaLocalVariable, JavaInterface, JavaEnum, JavaEnumConstant,
    GraphObject # Ensure GraphObject is imported if needed for isinstance checks, though type() is used later
)
from src.utils.java_parser import JavaParser, walk, find_node_by_type, extract_text, parse_package_declaration, find_node_by_type_in_children

class CodeExplanation(BaseModel):
    description: str = Field(..., description="자바 코드의 상세 설명 (한국어)")
    summary:     str = Field(..., description="1-2줄 요약 (한국어)")

class JavaAnalyzer:
    """Service for analyzing Java source code and building graph database"""
    
    NODE_TYPE_MAPPING = {
        JavaInternalPackage: {"label": "JavaInternalPackage", "pk": "name"},
        JavaLeafPackage: {"label": "JavaLeafPackage", "pk": "name"},
        JavaClass: {"label": "JavaClass", "pk": "name"},
        JavaInterface: {"label": "JavaInterface", "pk": "name"},
        JavaEnum: {"label": "JavaEnum", "pk": "name"},
        JavaEnumConstant: {"label": "JavaEnumConstant", "pk": "constant"},
        JavaField: {"label": "JavaField", "pk": "name"},
        JavaMethod: {"label": "JavaMethod", "pk": "signature"},
        JavaParameter: {"label": "JavaParameter", "pk": "name"},
        JavaLocalVariable: {"label": "JavaLocalVariable", "pk": "name"},
    }

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, openai_api_key: str):
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
    
    def _get_explanation_code(self, vertex_info: dict) -> Tuple[str, str]:
        """Get explanation of the code based on vertex_info dictionary.
           vertex_info is expected to have a 'body' key with the code string.
        """
        body = vertex_info.get("body")
        node_name = vertex_info.get("name", "Unnamed Node") # For logging/error messages
        node_labels = vertex_info.get("labels", [])

        additional_context = ""
        is_package_type = "JavaLeafPackage" in node_labels or "JavaInternalPackage" in node_labels

        # Escape curly braces in user-provided content to prevent Langchain from treating them as variables
        escaped_body = body.replace("{", "{{").replace("}", "}}") if body and isinstance(body, str) else None

        if "JavaClass" in node_labels:
            raw_class_context = self._build_incoming_context_for_class(vertex_info)
            additional_context = raw_class_context.replace("{", "{{").replace("}", "}}")
        elif is_package_type:
            raw_package_context = self._build_incoming_context_for_package(vertex_info)
            additional_context = raw_package_context.replace("{", "{{").replace("}", "}}")

        # Prepare variables for the prompt
        prompt_variables = {
            "format_instructions": self.llm_parser.get_format_instructions()
        }

        user_prompt_template_parts = []

        if not is_package_type and escaped_body and escaped_body.strip():
            user_prompt_template_parts.append("### Java 코드\n{code}\n\n")
            prompt_variables["code"] = escaped_body # Use escaped_body for {code}
        elif is_package_type and not (escaped_body and escaped_body.strip()): # Package type, body is not expected to be primary content
            prompt_variables["code"] = "" # Provide empty string for {code} if not available but template expects it
        elif not (escaped_body and escaped_body.strip()) and not (additional_context and additional_context.strip()):
             return (
                    f"{node_name}에 대해 LLM에 전달할 유효한 내용(코드 또는 컨텍스트)이 없습니다.",
                    "설명 생성 불가 (LLM 입력 내용 없음)"
                )
        else: # Non-package type without body, but has context. Or other unhandled cases.
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
                    "사용자가 제공한 내용을 읽고, 어떤 역할을 하는지 친절히 설명하고, "
                    "마지막에 한-두 줄로 핵심을 요약하세요. "
                    "주어진 Java 코드, 추가 컨텍스트가 있다면 그것들을 모두 고려하여 설명해주세요. "
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

        tx = self.graph.begin() # Use a transaction for read operations as well for consistency
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
    
    def get_or_create_package(self, package_name: str, is_leaf: bool = False) -> JavaPackage:
        """Get existing package or create new one. Packages have empty summary/description initially."""
        # Use a generic cache key for package name, specific type handled by instance check
        cache_key = (JavaPackage, package_name) 
        if cache_key in self.node_cache:
            package = self.node_cache[cache_key]
            # Check if the cached package's type is compatible with is_leaf expectation
            if is_leaf and not isinstance(package, JavaLeafPackage):
                print(f"Warning: Package {package_name} (cached as {type(package).__name__}) requested as JavaLeafPackage.")
            elif not is_leaf and not isinstance(package, JavaInternalPackage):
                print(f"Warning: Package {package_name} (cached as {type(package).__name__}) requested as JavaInternalPackage.")
            return package

        package = JavaPackage.match(self.graph, package_name).first()
        if not package:
            if is_leaf:
                # JavaLeafPackage __init__ expects name, description="", summary=""
                package = JavaLeafPackage(package_name, description="", summary="")
            else:
                # JavaInternalPackage __init__ expects name, description="", summary=""
                package = JavaInternalPackage(package_name, description="", summary="")
            self._add_to_batch(node=package)
        # If package exists, ensure it's the correct type or handle type mismatch if necessary
        # For now, we assume the first created type is correct.
        # Consider adding logic to merge/update if a package is found but type is different than expected.
        elif is_leaf and not isinstance(package, JavaLeafPackage):
            # This case should ideally be handled by the merge_duplicate_packages logic
            # or by ensuring consistent creation. For now, we'll just log it.
            print(f"Warning: Package {package_name} exists as {type(package).__name__} but requested as JavaLeafPackage.")
            # Optionally, update to JavaLeafPackage if it's currently a generic JavaPackage
        elif not is_leaf and not isinstance(package, JavaInternalPackage):
            print(f"Warning: Package {package_name} exists as {type(package).__name__} but requested as JavaInternalPackage.")
        
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
    
    def get_or_create_field(self, field_name: str, type: str) -> JavaField:
        """Get existing field or create new one"""
        cache_key = (JavaField, field_name) # Assuming field_name is globally unique for JavaField nodes
        if cache_key in self.node_cache:
            return self.node_cache[cache_key]

        field = JavaField.match(self.graph, field_name).first()
        if not field:
            field = JavaField(field_name, type)
            self._add_to_batch(node=field)
        
        if field: # Ensure field is not None before caching
             self.node_cache[cache_key] = field
        return field
    
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

    def get_or_create_parameter(self, parameter_name: str, type: str) -> JavaParameter:
        """Get existing parameter or create new one"""
        cache_key = (JavaParameter, parameter_name) # Assuming parameter_name is globally unique for JavaParameter nodes
        if cache_key in self.node_cache:
            return self.node_cache[cache_key]
            
        parameter = JavaParameter.match(self.graph, parameter_name).first()
        if not parameter:
            parameter = JavaParameter(parameter_name, type)
            self._add_to_batch(node=parameter)
        
        if parameter: # Ensure parameter is not None before caching
            self.node_cache[cache_key] = parameter
        return parameter
    
    def get_or_create_local_variable(self, local_variable_name: str, type: str) -> JavaLocalVariable:
        """Get existing local variable or create new one"""
        cache_key = (JavaLocalVariable, local_variable_name) # Assuming local_variable_name is globally unique
        if cache_key in self.node_cache:
            return self.node_cache[cache_key]

        local_variable = JavaLocalVariable.match(self.graph, local_variable_name).first()
        if not local_variable:
            local_variable = JavaLocalVariable(local_variable_name, type)
            self._add_to_batch(node=local_variable)
        
        if local_variable: # Ensure local_variable is not None before caching
            self.node_cache[cache_key] = local_variable
        return local_variable
    
    def get_or_create_enum_constant(self, constant: str) -> JavaEnumConstant:
        """Get existing enum constant or create new one"""
        cache_key = (JavaEnumConstant, constant) # `constant` is the primary key
        if cache_key in self.node_cache:
            return self.node_cache[cache_key]

        enum_constant = JavaEnumConstant.match(self.graph, constant).first()
        if not enum_constant:
            enum_constant = JavaEnumConstant(constant)
            self._add_to_batch(node=enum_constant)
        
        if enum_constant: # Ensure enum_constant is not None before caching
            self.node_cache[cache_key] = enum_constant
        return enum_constant
    
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

    def build_package_hierarchy(self, package_path: str) -> Tuple[Optional[JavaPackage], Optional[JavaPackage]]:
        """Build package hierarchy from package path"""
        if not package_path:
            return None, None
            
        parts = package_path.split('.')
        root_package = None
        before_package = None
        leaf_package = None

        for i, part in enumerate(parts):
            current_package_name = '.'.join(parts[:i+1])
            
            try:
                if i == len(parts) - 1:
                    leaf_package = self.get_or_create_package(current_package_name, is_leaf=True)
                    if before_package:
                        leaf_package.package_by.add(before_package)
                        self._add_to_batch(node=leaf_package)
                else:
                    current_package = self.get_or_create_package(current_package_name)
                    if root_package is None:
                        root_package = current_package
                        before_package = current_package
                        self._add_to_batch(node=current_package)
                    else:
                        current_package.package_by.add(before_package)
                        self._add_to_batch(node=current_package)
                        before_package = current_package
            except Exception as e:
                print(f"Error processing package {current_package_name}: {str(e)}")
                return None, None
        
        return root_package, leaf_package
    
    def process_imports(self, root: TreeSitterNode, leaf_package: JavaLeafPackage) -> None:
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
                
            # Build package hierarchy for the actual package
            imported_root_package, imported_package = self.build_package_hierarchy(package_name)
                
            if imported_package: # This should be a JavaPackage (either LeafPackage or InternalPackage)
                if isinstance(imported_package, JavaPackage):
                    # Create import relationship between packages
                    leaf_package.imported_by.add(imported_package)
                    self._add_to_batch(node=leaf_package) # Mark leaf_package as needing update
                    self._add_to_batch(node=imported_package) # Mark imported package as needing update

    
    def find_field_declaration(self, node: TreeSitterNode) -> JavaField:
        """Extract field declaration information"""
        type_name = ""
        variable_name = ""
        for child in walk(node):
            if child.type == "type_identifier":
                type_name = extract_text(child)
            if child.type == "variable_declarator":
                variable_name = extract_text(child)
        return self.get_or_create_field(variable_name, type_name)
    
    def find_local_variable_declaration(self, node: TreeSitterNode) -> JavaLocalVariable:
        """Extract local variable declaration information"""
        type_name = ""
        variable_name = ""
        for child_node in walk(node): # Renamed child to child_node
            if child_node.type == "type_identifier":
                type_name = extract_text(child_node)
            if child_node.type == "variable_declarator":
                # Ensure there's a child node before accessing its children
                if child_node.children:
                    variable_name = extract_text(child_node.children[0])
                else: # Handle cases like "int x;" where declarator itself is the name
                    variable_name = extract_text(child_node)

        return self.get_or_create_local_variable(variable_name, type_name)
    
    def find_formal_parameter(self, node: TreeSitterNode) -> JavaParameter:
        """Extract formal parameter information"""
        type_name = ""
        variable_name = ""
        for child in walk(node):
            if child.type == "formal_parameter":
                list_split = extract_text(child).split(" ")
                type_name = list_split[0].strip()
                variable_name = list_split[1].strip()
                break
        return self.get_or_create_parameter(variable_name, type_name)
    
    def find_extended_class(self, node: TreeSitterNode) -> List[JavaClass]:
        """Extract extended class information"""
        extended_class = None
        # tree-sitter node for superclass is "superclass"
        # its child is "type_identifier" which contains the name
        type_identifier_node = find_node_by_type_in_children(node, "type_identifier")
        if type_identifier_node:
            extended_class_name = extract_text(type_identifier_node)
            extended_class = self.get_or_create_class(extended_class_name) # Body will be empty for now if not in project
        return extended_class
    
    def find_implemented_interface(self, node: TreeSitterNode) -> JavaInterface:
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

    def _process_class_node(self, class_node: TreeSitterNode, leaf_package: JavaLeafPackage) -> None:
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
    
    def _process_interface_node(self, interface_node: TreeSitterNode, leaf_package: JavaLeafPackage) -> None:
        """Process a interface declaration node"""
        body = extract_text(interface_node)
        name_node = find_node_by_type_in_children(interface_node, "identifier")
        if not name_node:
            return

        interface_name = extract_text(name_node)
        java_interface = self.get_or_create_interface(interface_name, body=body, description="", summary="")

        java_interface.contained_in.add(leaf_package)
        self._add_to_batch(node=java_interface)

    def _process_enum_node(self, enum_node: TreeSitterNode, leaf_package: JavaLeafPackage) -> None:
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

        elif class_child_node.type == "field_declaration" and class_child_node.parent == find_node_by_type_in_children(parent_class_node, "class_body"): # More specific parent check
            field = self.find_field_declaration(class_child_node)
            if field and field.name: # Ensure field and its name are valid
                java_class.fields.add(field)
                self._add_to_batch(node=java_class) # Mark for update

        elif class_child_node.type == "method_declaration" and class_child_node.parent == find_node_by_type_in_children(parent_class_node, "class_body"): # More specific parent check
            self._process_method_node(class_child_node, java_class)

    def _process_interface_child_node(self, interface_child_node: TreeSitterNode, java_interface: JavaInterface) -> None:
        # Example: processing constant fields or default methods in interfaces
        if interface_child_node.type == "method_declaration": # default or static methods in interfaces
             self._process_method_node(interface_child_node, java_interface) # Pass java_interface
        # Add more processing for other interface elements if needed (e.g., constant_declaration)

    def _process_enum_child_node(self, enum_child_node: TreeSitterNode, java_enum: JavaEnum) -> None:
        if enum_child_node.type == "enum_constant":
            # An enum_constant should be a child of enum_body, which is a child of enum_declaration.
            # The `java_enum` object corresponds to the parent enum_declaration node.
            
            # Check proper nesting and that the constant belongs to the current java_enum
            if (enum_child_node.parent and
                enum_child_node.parent.type == "enum_body" and
                enum_child_node.parent.parent and
                enum_child_node.parent.parent.type == "enum_declaration"):
                
                # Ensure this constant belongs to the specific enum (java_enum) being processed.
                # This is done by checking if the name of the grandparent enum_declaration matches java_enum.name.
                grandparent_enum_decl_node = enum_child_node.parent.parent
                identifier_of_grandparent_enum = find_node_by_type_in_children(grandparent_enum_decl_node, "identifier")

                if identifier_of_grandparent_enum and extract_text(identifier_of_grandparent_enum) == java_enum.name:
                    constant_name_node = find_node_by_type_in_children(enum_child_node, "identifier")
                    if constant_name_node:
                        constant = extract_text(constant_name_node)
                        java_enum_constant = self.get_or_create_enum_constant(constant)
                        java_enum.constants.add(java_enum_constant)
                        self._add_to_batch(node=java_enum_constant)
                        self._add_to_batch(node=java_enum)

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
        
        for param_node_container in walk(method_node): 
            if param_node_container.type == "formal_parameters":
                for param_node in param_node_container.children: # Iterate actual formal_parameter nodes
                    if param_node.type == "formal_parameter":
                        java_parameter = self.find_formal_parameter(param_node) # Pass the 'formal_parameter' node
                        if java_parameter and java_parameter.name and java_parameter.type : # Ensure valid parameter
                            self._add_to_batch(node=java_parameter)
                            java_method.parameters.add(java_parameter)
                            self._add_to_batch(node=java_method) # Mark method for update

            elif param_node_container.type == "local_variable_declaration":
                # Ensure local_variable_declaration is within the method_body
                method_body_node = find_node_by_type_in_children(method_node, "block") # Assuming block is method body
                if method_body_node and param_node_container.parent == method_body_node:
                    java_local_variable = self.find_local_variable_declaration(param_node_container)
                    if java_local_variable and java_local_variable.name and java_local_variable.type: # Ensure valid local var
                        self._add_to_batch(node=java_local_variable)
                        java_method.local_variables.add(java_local_variable)
                        self._add_to_batch(node=java_method) # Mark method for update
        
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

            _, leaf_package = self.build_package_hierarchy(package_name)
            if not leaf_package:
                print(f"Warning: Could not build/get leaf package for {package_name} in {file_path}. Skipping.")
                return
            
            # Ensure leaf_package is indeed a JavaLeafPackage as expected by process_imports and others
            if not isinstance(leaf_package, JavaLeafPackage):
                print(f"Warning: Expected JavaLeafPackage for {package_name} but got {type(leaf_package)}. Attempting to proceed.")
                # This might indicate an issue in build_package_hierarchy or get_or_create_package if a non-leaf
                # was returned where a leaf was expected. Or, if an internal package was misidentified as leaf.

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
        """Merge JavaLeafPackage into JavaInternalPackage if they share the same name."""
        tx = self.graph.begin()
        try:
            query = (
                "MATCH (leaf:JavaLeafPackage), (internal:JavaInternalPackage) "
                "WHERE leaf.name = internal.name "
                "RETURN leaf, internal"
            )
            data = tx.run(query)

            for record in data:
                leaf_package = record["leaf"]
                internal_package = record["internal"]

                # Transfer incoming relationships (PARENT_PACKAGE, IMPORT)
                # PARENT_PACKAGE from child to leaf -> child to internal
                transfer_parent_query = (
                    "MATCH (child)-[r:PARENT_PACKAGE]->(leaf:JavaLeafPackage {name: $leaf_name}) "
                    "MATCH (internal:JavaInternalPackage {name: $internal_name}) "
                    "CREATE (child)-[:PARENT_PACKAGE]->(internal)"
                )
                tx.run(transfer_parent_query, leaf_name=leaf_package["name"], internal_name=internal_package["name"])
                
                # IMPORT from other_package to leaf -> other_package to internal
                transfer_import_to_leaf_query = (
                    "MATCH (other)-[r:IMPORT]->(leaf:JavaLeafPackage {name: $leaf_name}) "
                    "MATCH (internal:JavaInternalPackage {name: $internal_name}) "
                    "CREATE (other)-[:IMPORT]->(internal)"
                )
                tx.run(transfer_import_to_leaf_query, leaf_name=leaf_package["name"], internal_name=internal_package["name"])

                # Transfer outgoing relationships (IMPORT, CLASS, INTERFACE, ENUM, PARENT_PACKAGE of leaf)
                # IMPORT from leaf to other_package -> internal to other_package
                transfer_import_from_leaf_query = (
                    "MATCH (leaf:JavaLeafPackage {name: $leaf_name})-[r:IMPORT]->(other) "
                    "MATCH (internal:JavaInternalPackage {name: $internal_name}) "
                    "CREATE (internal)-[:IMPORT]->(other)"
                )
                tx.run(transfer_import_from_leaf_query, leaf_name=leaf_package["name"], internal_name=internal_package["name"])
                
                # PARENT_PACKAGE from leaf to its parent_pkg -> internal to parent_pkg
                transfer_leafs_parent_query = (
                    "MATCH (leaf:JavaLeafPackage {name: $leaf_name})-[:PARENT_PACKAGE]->(parent_pkg) "
                    "MATCH (internal:JavaInternalPackage {name: $internal_name}) "
                    "MERGE (internal)-[:PARENT_PACKAGE]->(parent_pkg)"
                )
                tx.run(transfer_leafs_parent_query, leaf_name=leaf_package["name"], internal_name=internal_package["name"])
                
                # CLASS from leaf to class_node -> internal to class_node
                transfer_class_query = (
                    "MATCH (leaf:JavaLeafPackage {name: $leaf_name})<-[:CLASS]-(class_node) "
                    "MATCH (internal:JavaInternalPackage {name: $internal_name}) "
                    "CREATE (internal)<-[:CLASS]-(class_node)"
                )
                tx.run(transfer_class_query, leaf_name=leaf_package["name"], internal_name=internal_package["name"])
                
                # INTERFACE from leaf to interface_node -> internal to interface_node
                transfer_interface_query = (
                    "MATCH (leaf:JavaLeafPackage {name: $leaf_name})<-[:INTERFACE]-(interface_node) "
                    "MATCH (internal:JavaInternalPackage {name: $internal_name}) "
                    "CREATE (internal)<-[:INTERFACE]-(interface_node)"
                )
                tx.run(transfer_interface_query, leaf_name=leaf_package["name"], internal_name=internal_package["name"])
                
                # ENUM from leaf to enum_node -> internal to enum_node
                transfer_enum_query = (
                    "MATCH (leaf:JavaLeafPackage {name: $leaf_name})<-[:ENUM]-(enum_node) "
                    "MATCH (internal:JavaInternalPackage {name: $internal_name}) "
                    "CREATE (internal)<-[:ENUM]-(enum_node)"
                )
                tx.run(transfer_enum_query, leaf_name=leaf_package["name"], internal_name=internal_package["name"])

                # Delete the JavaLeafPackage node and its relationships
                # Detach delete ensures relationships are removed before deleting the node.
                delete_leaf_query = (
                    "MATCH (leaf:JavaLeafPackage {name: $leaf_name}) "
                    "DETACH DELETE leaf"
                )
                tx.run(delete_leaf_query, leaf_name=leaf_package["name"])

            self.graph.commit(tx)
            print("Duplicate packages merged successfully.")

        except Exception as e:
            if tx:
                self.graph.rollback(tx)
            print(f"Error merging duplicate packages: {str(e)}")

    def collect_all_analyzed_vertices(self) -> List[dict]:
        """Collects all vertices with specified Java-related labels from the graph."""
        print("Collecting all analyzed vertices (JavaMethod, JavaClass, etc.)...")
        collected_vertices = []
        labels_to_collect = [
            "JavaMethod", "JavaClass", "JavaEnum", "JavaInterface",
            "JavaInternalPackage", "JavaLeafPackage"
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
                    "MATCH (p:JavaLeafPackage) "
                    "WHERE p.name =~ $pattern "
                    "RETURN p.name AS name"
                )
                
                result = tx.run(query, pattern=pattern)
                incorrect_packages = [record["name"] for record in result]
                
                # Delete these incorrectly created package nodes
                for package_name in incorrect_packages:
                    print(f"Deleting incorrectly created package: {package_name}")
                    delete_query = (
                        "MATCH (p:JavaLeafPackage {name: $package_name}) "
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
                    "MATCH (p:JavaLeafPackage {name: $class_name}) "
                    "RETURN count(p) as count"
                )
                result = tx.run(check_query, class_name=class_name)
                
                # More robust way to get result
                try:
                    records = list(result)
                    if records and records[0]["count"] > 0:
                        print(f"Deleting incorrectly created package: {class_name}")
                        delete_query = (
                            "MATCH (p:JavaLeafPackage {name: $class_name}) "
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
                "CREATE CONSTRAINT JavaMethod_signature IF NOT EXISTS FOR (m:JavaMethod) REQUIRE m.signature IS UNIQUE",
                "CREATE CONSTRAINT JavaClass_name IF NOT EXISTS FOR (c:JavaClass) REQUIRE c.name IS UNIQUE",
                "CREATE CONSTRAINT JavaInterface_name IF NOT EXISTS FOR (i:JavaInterface) REQUIRE i.name IS UNIQUE",
                "CREATE CONSTRAINT JavaEnum_name IF NOT EXISTS FOR (e:JavaEnum) REQUIRE e.name IS UNIQUE",
                "CREATE CONSTRAINT JavaField_name IF NOT EXISTS FOR (f:JavaField) REQUIRE f.name IS UNIQUE",
                "CREATE CONSTRAINT JavaParameter_name IF NOT EXISTS FOR (p:JavaParameter) REQUIRE p.name IS UNIQUE",
                "CREATE CONSTRAINT JavaLocalVariable_name IF NOT EXISTS FOR (v:JavaLocalVariable) REQUIRE v.name IS UNIQUE",
                "CREATE CONSTRAINT JavaEnumConstant_constant IF NOT EXISTS FOR (ec:JavaEnumConstant) REQUIRE ec.constant IS UNIQUE",
                "CREATE CONSTRAINT JavaInternalPackage_name IF NOT EXISTS FOR (ip:JavaInternalPackage) REQUIRE ip.name IS UNIQUE",
                "CREATE CONSTRAINT JavaLeafPackage_name IF NOT EXISTS FOR (lp:JavaLeafPackage) REQUIRE lp.name IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    self.graph.run(constraint)
                except Exception as e:
                    print(f"Error creating constraint: {constraint}, Error: {e}")
            
            print("Database schema initialized successfully.")
            
        except Exception as e:
            print(f"Error initializing database schema: {e}")
            raise


def find_java_files(directory: str) -> List[str]:
    """Find all Java files in directory"""
    java_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.java'):
                java_files.append(os.path.join(root, file))
    return java_files 