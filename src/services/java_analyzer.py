import os
import sys
from typing import List, Tuple, Optional
import uuid # Added for unique IDs
import json # Added for JSON operations
import time # Added for polling sleep
import openai # Added for OpenAI Batch API
from tqdm import tqdm
from py2neo import Graph, Node, Relationship
from tree_sitter import Node
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI


# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.java_models import (
    JavaPackage, JavaInternalPackage, JavaLeafPackage,
    JavaClass, JavaMethod, JavaField, JavaParameter, JavaLocalVariable, JavaInterface, JavaEnum, JavaEnumConstant
)
from src.utils.java_parser import JavaParser, walk, find_node_by_type, extract_text, parse_package_declaration, find_node_by_type_in_children

# .env 파일에서 환경변수 로드


# OpenAI API 키 확인


# Pydantic 모델 정의
class CodeExplanation(BaseModel):
    description: str = Field(..., description="자바 코드의 상세 설명 (한국어)")
    summary:     str = Field(..., description="1-2줄 요약 (한국어)")

class JavaAnalyzer:
    """Service for analyzing Java source code and building graph database"""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, openai_api_key: str):
        self.graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.parser = JavaParser()
        self.batch_size = 1000
        self.pending_nodes = []
        self.pending_relationships = []
        self.snippets_for_explanation = [] # Added for accumulating code snippets
        self.openai_api_key = openai_api_key # Store for client initialization
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key) # Initialize OpenAI client
        self.llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=openai_api_key)  # temperature를 0으로 설정하여 일관된 출력
        self.llm_parser = PydanticOutputParser(pydantic_object=CodeExplanation)
    
    async def _batch_explain_code_snippets(self) -> None:
        """Processes accumulated code snippets using the OpenAI Batch API."""
        if not self.snippets_for_explanation:
            print("No snippets to explain.")
            return

        batch_input_file_path = "batch_input.jsonl"
        custom_id_to_node_map = {}
        batch_requests = []

        # Prepare format instructions once
        format_instructions = self.llm_parser.get_format_instructions()
        
        # System message for the prompt
        system_message_content = (
            "당신은 숙련된 자바 개발자입니다. "
            "사용자가 준 Java 코드를 읽고, 어떤 역할을 하는지 친절히 설명하고, "
            "마지막에 한-두 줄로 핵심을 요약하세요. "
            f"반드시 제공된 JSON 스키마에 맞춰 응답하세요.\n{format_instructions}"
        )

        for i, (graph_node, body_text) in enumerate(self.snippets_for_explanation):
            custom_id = f"request-{uuid.uuid4()}" # More robust unique ID
            custom_id_to_node_map[custom_id] = graph_node
            
            request_body = {
                "model": "gpt-4", # Or self.llm.model_name if accessible and desired
                "messages": [
                    {"role": "system", "content": system_message_content},
                    {"role": "user", "content": f"### Java 코드\n{body_text}\n\n### 출력 형식 지침\n{format_instructions}"} 
                ],
                "temperature": 0, # Or self.llm.temperature if accessible
                "response_format": {"type": "json_object"},
            }
            
            batch_requests.append({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": request_body
            })

        try:
            # Create and Upload Batch File
            with open(batch_input_file_path, "w", encoding="utf-8") as f:
                for req in batch_requests:
                    f.write(json.dumps(req) + "\n")
            
            batch_input_file = self.openai_client.files.create(
                file=open(batch_input_file_path, "rb"),
                purpose="batch"
            )
            os.remove(batch_input_file_path) # Clean up local file

            # Create Batch Job
            batch_job = self.openai_client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            print(f"Batch job created with ID: {batch_job.id}")

            # Poll for Completion
            start_time = time.time()
            timeout_seconds = 3600  # 1 hour timeout for the entire batch job
            polling_interval_seconds = 30 
            max_retries = timeout_seconds // polling_interval_seconds

            for i in range(max_retries):
                batch_job = self.openai_client.batches.retrieve(batch_job.id)
                print(f"Polling batch job status: {batch_job.status} (Attempt {i+1}/{max_retries})")
                
                if batch_job.status == 'completed':
                    break
                elif batch_job.status in ['failed', 'cancelled']:
                    print(f"Batch job {batch_job.status}. Errors: {batch_job.errors}")
                    self.snippets_for_explanation.clear() # Clear snippets even on failure
                    return
                
                if time.time() - start_time > timeout_seconds:
                    print("Batch job polling timed out.")
                    self.openai_client.batches.cancel(batch_job.id) # Attempt to cancel
                    self.snippets_for_explanation.clear()
                    return
                
                time.sleep(polling_interval_seconds)
            else: # Executed if the loop completes without break (max_retries reached)
                print(f"Batch job did not complete after {max_retries} retries. Last status: {batch_job.status}")
                if batch_job.status != 'completed': # Check if not completed despite loop finishing
                   try:
                       self.openai_client.batches.cancel(batch_job.id)
                       print(f"Attempted to cancel batch job {batch_job.id}")
                   except Exception as e_cancel:
                       print(f"Error cancelling batch job {batch_job.id}: {e_cancel}")
                self.snippets_for_explanation.clear()
                return

            # Process Results
            if batch_job.status == 'completed' and batch_job.output_file_id:
                output_file_id = batch_job.output_file_id
                result_content_response = self.openai_client.files.content(output_file_id)
                result_content = result_content_response.read().decode('utf-8')
                
                print("Batch job completed. Processing results...")
                results_data = result_content.strip().split("\n")

                for line in results_data:
                    try:
                        result_json = json.loads(line)
                        custom_id = result_json.get("custom_id")
                        graph_node = custom_id_to_node_map.get(custom_id)

                        if not graph_node:
                            print(f"Warning: No graph node found for custom_id {custom_id}")
                            continue
                        
                        response_body = result_json.get("response", {}).get("body", {})
                        if not response_body:
                            print(f"Warning: Empty response body for custom_id {custom_id}")
                            # Set default error message or skip
                            graph_node.description = "Error processing snippet (empty response body)."
                            graph_node.summary = "Processing error."
                            self._add_to_batch(node=graph_node)
                            continue

                        message_content_str = response_body.get("choices", [{}])[0].get("message", {}).get("content")
                        
                        if not message_content_str:
                            print(f"Warning: No message content for custom_id {custom_id}")
                            graph_node.description = "Error processing snippet (no content)."
                            graph_node.summary = "Processing error."
                            self._add_to_batch(node=graph_node)
                            continue

                        # Parse the JSON string content into CodeExplanation model
                        explanation = CodeExplanation.model_validate_json(message_content_str)
                        graph_node.description = explanation.description
                        graph_node.summary = explanation.summary
                        self._add_to_batch(node=graph_node) # Add to Neo4j batch
                        print(f"Updated node for custom_id {custom_id}")

                    except json.JSONDecodeError as e_json:
                        print(f"Error decoding JSON from result line: {line}. Error: {e_json}")
                    except Exception as e_proc:
                        print(f"Error processing result line {line}. Error: {e_proc}")
                
                print("Finished processing batch results.")
            else:
                print(f"Batch job completed but no output file ID or status is not 'completed'. Status: {batch_job.status}")

        except openai.APIError as e:
            print(f"OpenAI API error: {e}")
        except IOError as e:
            print(f"File IO error: {e}")
            if os.path.exists(batch_input_file_path): # Ensure cleanup if file still exists
                os.remove(batch_input_file_path)
        except Exception as e:
            print(f"An unexpected error occurred in _batch_explain_code_snippets: {e}")
        finally:
            self.snippets_for_explanation.clear() # Always clear the list
            print("Snippets explanation list cleared.")


    def _get_explanation_code(self, body: str) -> Tuple[str, str]:
        """Get explanation of the code"""
        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system",
                    "당신은 숙련된 자바 개발자입니다. "
                    "사용자가 준 Java 코드를 읽고, 어떤 역할을 하는지 친절히 설명하고, "
                    "마지막에 한-두 줄로 핵심을 요약하세요. "
                    "반드시 제공된 JSON 스키마에 맞춰 응답하세요."),
                    ("user",
                    "### Java 코드\n{code}\n\n"
                    "### 출력 형식 지침\n{format_instructions}")
                ]
            )


            chain: Runnable = prompt | self.llm | self.llm_parser
            result: CodeExplanation = chain.invoke(
                {
                    "code": body,
                    "format_instructions": self.llm_parser.get_format_instructions(),
                }
            )
            
            # 결과가 None이거나 필수 필드가 없는 경우 기본값 사용
            if not result or not result.description or not result.summary:
                return (
                    "이 코드는 Java 클래스/인터페이스의 구현입니다.",
                    "Java 코드 분석 중 오류가 발생했습니다."
                )
                
            return result.description, result.summary
            
        except Exception as e:
            print(f"Error analyzing code: {str(e)}")
            return (
                "이 코드는 Java 클래스/인터페이스의 구현입니다.",
                "Java 코드 분석 중 오류가 발생했습니다."
            )
    
    
    def _flush_batch(self):
        """Flush pending nodes and relationships to Neo4j"""
        if not self.pending_nodes and not self.pending_relationships:
            return

        tx = self.graph.begin()
        try:
            # 1) 노드(혹은 GraphObject)
            for obj in self.pending_nodes:
                if isinstance(obj, (Node, Relationship)):
                    tx.create(obj)          # Low-level 객체는 그대로
                else:
                    tx.merge(obj)           # OGM 객체는 merge

            # 2) 관계
            for rel in self.pending_relationships:
                tx.create(rel)

            self.graph.commit(tx)
        except Exception as e:
            self.graph.rollback(tx)
            raise e
        finally:
            self.pending_nodes.clear()
            self.pending_relationships.clear()
    
    def _add_to_batch(self, node=None, relationship=None):
        """Add node or relationship to pending batch"""
        if node:
            self.pending_nodes.append(node)
        if relationship:
            self.pending_relationships.append(relationship)
            
        if len(self.pending_nodes) + len(self.pending_relationships) >= self.batch_size:
            self._flush_batch()
    
    def get_or_create_package(self, package_name: str, is_leaf: bool = False) -> JavaPackage:
        """Get existing package or create new one"""
        package = JavaPackage.match(self.graph, package_name).first()
        if not package:
            package = JavaLeafPackage(package_name) if is_leaf else JavaInternalPackage(package_name)
            self._add_to_batch(node=package)
        return package
    
    def get_or_create_class(self, class_name: str, body: str = "", description: str = "", summary: str = "") -> JavaClass:
        """Get existing class or create new one"""
        cls = JavaClass.match(self.graph, class_name).first()
        if not cls:
            cls = JavaClass(class_name, body, description, summary)
            self._add_to_batch(node=cls)
        return cls
    
    def get_or_create_interface(self, interface_name: str, body: str = "", description: str = "", summary: str = "") -> JavaInterface:
        """Get existing interface or create new one"""
        interface = JavaInterface.match(self.graph, interface_name).first()
        if not interface:
            interface = JavaInterface(interface_name, body, description, summary)
            self._add_to_batch(node=interface)
        return interface
    
    def get_or_create_field(self, field_name: str, type: str) -> JavaField:
        """Get existing field or create new one"""
        field = JavaField.match(self.graph, field_name).first()
        if not field:
            field = JavaField(field_name, type)
            self._add_to_batch(node=field)
        return field
    
    def get_or_create_method(self, method_name: str, body: str = "", description: str = "", return_type: str = "", summary: str = "") -> JavaMethod:
        """Get existing method or create new one"""
        method = JavaMethod.match(self.graph, method_name).first()
        if not method:
            method = JavaMethod(method_name, body, description, summary, return_type)
            self._add_to_batch(node=method)
        return method

    def get_or_create_parameter(self, parameter_name: str, type: str) -> JavaParameter:
        """Get existing parameter or create new one"""
        parameter = JavaParameter.match(self.graph, parameter_name).first()
        if not parameter:
            parameter = JavaParameter(parameter_name, type)
            self._add_to_batch(node=parameter)
        return parameter
    
    def get_or_create_local_variable(self, local_variable_name: str, type: str) -> JavaLocalVariable:
        """Get existing local variable or create new one"""
        local_variable = JavaLocalVariable.match(self.graph, local_variable_name).first()
        if not local_variable:
            local_variable = JavaLocalVariable(local_variable_name, type)
            self._add_to_batch(node=local_variable)
        return local_variable
    
    def get_or_create_enum_constant(self, constant: str) -> JavaEnumConstant:
        """Get existing enum constant or create new one"""
        enum_constant = JavaEnumConstant.match(self.graph, constant).first()
        if not enum_constant:
            enum_constant = JavaEnumConstant(constant)
            self._add_to_batch(node=enum_constant)
        return enum_constant
    
    def get_or_create_enum(self, enum_name: str, body: str = "", description: str = "", summary: str = "") -> JavaEnum:
        """Get existing enum or create new one"""
        enum = JavaEnum.match(self.graph, enum_name).first()
        if not enum:
            enum = JavaEnum(enum_name, body, description, summary)
            self._add_to_batch(node=enum)
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
    
    def process_imports(self, root: Node, leaf_package: JavaLeafPackage) -> None:
        """Process import declarations"""
        for imp in (n for n in root.children if n.type == "import_declaration"):
            imp_text = extract_text(imp).strip()
            is_static = " static " in imp_text
            path = imp_text.replace("import", "").replace(";", "").strip()
            
            if is_static:
                fq = path.split(" ", 1)[1]
                root_package_tmp, leaf_package_tmp = self.build_package_hierarchy(fq)
            elif path.endswith(".*"):
                root_package_tmp, leaf_package_tmp = self.build_package_hierarchy(path[:-2])
            else:
                root_package_tmp, leaf_package_tmp = self.build_package_hierarchy(path)
                
            if leaf_package_tmp:
                leaf_package_tmp.imports.add(leaf_package)
                leaf_package.imported_by.add(leaf_package_tmp)
                self._add_to_batch(node=leaf_package_tmp)
                self._add_to_batch(node=leaf_package)
    
    def find_field_declaration(self, node: Node) -> JavaField:
        """Extract field declaration information"""
        type_name = ""
        variable_name = ""
        for child in walk(node):
            if child.type == "type_identifier":
                type_name = extract_text(child)
            if child.type == "variable_declarator":
                variable_name = extract_text(child)
        return self.get_or_create_field(variable_name, type_name)
    
    def find_local_variable_declaration(self, node: Node) -> JavaLocalVariable:
        """Extract local variable declaration information"""
        type_name = ""
        variable_name = ""
        for child in walk(node):
            if child.type == "type_identifier":
                type_name = extract_text(child)
            if child.type == "variable_declarator":
                variable_name = extract_text(child.children[0])
        return self.get_or_create_local_variable(variable_name, type_name)
    
    def find_formal_parameter(self, node: Node) -> JavaParameter:
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
    
    def find_extended_class(self, node: Node) -> List[JavaClass]:
        """Extract extended class information"""
        extended_class = None
        for child in walk(node):
            if child.type == "type_identifier":
                extended_class_name = extract_text(child)
                extended_class = self.get_or_create_class(extended_class_name)
                break
        return extended_class
    
    def find_implemented_interface(self, node: Node) -> JavaInterface:
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

    def _process_class_node(self, class_node: Node, leaf_package: JavaLeafPackage) -> None:
        """Process a class declaration node"""
        body = extract_text(class_node)
        # description, summary = self._get_explanation_code(body) # Removed
        name_node = find_node_by_type_in_children(class_node, "identifier")
        if not name_node:
            return
        cls_name = extract_text(name_node)
        java_class = self.get_or_create_class(class_name=cls_name, body=body, description="", summary="")
        java_class.contained_in.add(leaf_package)
        self._add_to_batch(node=java_class)
        self.snippets_for_explanation.append((java_class, body)) # Added snippet

        for class_child_node in walk(class_node):
            self._process_class_child_node(class_child_node, java_class)
    
    def _process_interface_node(self, interface_node: Node, leaf_package: JavaLeafPackage) -> None:
        """Process a interface declaration node"""
        body = extract_text(interface_node)
        # description, summary = self._get_explanation_code(body) # Removed
        name_node = find_node_by_type_in_children(interface_node, "identifier")
        if not name_node:
            return

        interface_name = extract_text(name_node)
        java_interface = self.get_or_create_interface(interface_name, body=body, description="", summary="")
        java_interface.contained_in.add(leaf_package)
        self._add_to_batch(node=java_interface)
        self.snippets_for_explanation.append((java_interface, body)) # Added snippet

    def _process_enum_node(self, enum_node: Node, leaf_package: JavaLeafPackage) -> None:
        """Process a enum declaration node"""
        body = extract_text(enum_node)
        # description, summary = self._get_explanation_code(body) # Removed
        name_node = find_node_by_type_in_children(enum_node, "identifier")
        if not name_node:
            return

        enum_name = extract_text(name_node)
        java_enum = self.get_or_create_enum(enum_name, body=body, description="", summary="")
        java_enum.contained_in.add(leaf_package)
        self._add_to_batch(node=java_enum)
        self.snippets_for_explanation.append((java_enum, body)) # Added snippet
        
        for enum_child_node in walk(enum_node):
            self._process_enum_child_node(enum_child_node, java_enum)

    def _process_class_child_node(self, class_child_node: Node, java_class: JavaClass) -> None:
        """Process a child node of a class declaration"""
        if class_child_node.type == "superclass":
            extended_class = self.find_extended_class(class_child_node)
            java_class.extends.add(extended_class)
            self._add_to_batch(node=java_class)

        elif class_child_node.type == "super_interfaces":
            implemented_interfaces = self.find_implemented_interface(class_child_node)
            for implemented_interface in implemented_interfaces:
                java_class.implements.add(implemented_interface)
            self._add_to_batch(node=java_class)

        elif class_child_node.type == "field_declaration":
            field = self.find_field_declaration(class_child_node)
            java_class.fields.add(field)
            self._add_to_batch(node=java_class)

        elif class_child_node.type == "method_declaration":
            self._process_method_node(class_child_node, java_class)

    def _process_interface_child_node(self, interface_child_node: Node, java_interface: JavaInterface) -> None:
        pass

    def _process_enum_child_node(self, enum_child_node: Node, java_enum: JavaEnum) -> None:
        if enum_child_node.type == "enum_constant":
            constant = extract_text(enum_child_node)
            java_enum_constant = self.get_or_create_enum_constant(constant)
            java_enum.constants.add(java_enum_constant)
            self._add_to_batch(node=java_enum_constant)

    def _process_method_node(self, method_node: Node, java_class: JavaClass | JavaInterface) -> None:
        """Process a method declaration node"""
        body = extract_text(method_node)
        # description, summary = self._get_explanation_code(body) # Removed
        name_node = find_node_by_type_in_children(method_node, "identifier")
        if not name_node:
            return

        method_name = extract_text(name_node)
        return_type_node = find_node_by_type(method_node, "type_identifier")
        return_type = extract_text(return_type_node) if return_type_node else "void"

        # Note: JavaMethod is directly instantiated, not using a get_or_create pattern here that takes all params.
        # We need to ensure its direct instantiation handles description="" and summary="".
        # The JavaMethod class definition would need to have default "" for description and summary,
        # or be called as JavaMethod(method_name, body=body, description="", summary="", return_type=return_type)
        # Assuming JavaMethod model is defined like:
        # class JavaMethod(GraphObject):
        #   name = Property()
        #   body = Property()
        #   description = Property()
        #   summary = Property()
        #   return_type = Property()
        #   def __init__(self, name, body="", description="", summary="", return_type=""): ...
        # Then the call below is fine.
        java_method = JavaMethod(method_name, body=body, description="", summary="", return_type=return_type)
        self._add_to_batch(node=java_method)
        self.snippets_for_explanation.append((java_method, body)) # Added snippet

        for param_node in walk(method_node):
            if param_node.type == "formal_parameters":
                java_parameter = self.find_formal_parameter(param_node)
                self._add_to_batch(node=java_parameter)
                java_method.parameters.add(java_parameter)
                self._add_to_batch(node=java_method)

            elif param_node.type == "local_variable_declaration":
                java_local_variable = self.find_local_variable_declaration(param_node)
                self._add_to_batch(node=java_local_variable)
                java_method.local_variables.add(java_local_variable)
                self._add_to_batch(node=java_method)

        java_class.methods.add(java_method)
        self._add_to_batch(node=java_class)
    

    async def process_java_file(self, file_path: str) -> None:
        """Process a single Java file"""
        try:
            source_code = self._read_source_code(file_path)
            root = self.parser.parse(source_code)
            package_name = parse_package_declaration(root)

            if not package_name:
                return

            root_package, leaf_package = self.build_package_hierarchy(package_name)
            if not leaf_package:
                return

            self.process_imports(root, leaf_package)

            for node in walk(root):
                if node.type == "class_declaration":
                    self._process_class_node(node, leaf_package)
                elif node.type == "interface_declaration":
                    self._process_interface_node(node, leaf_package)
                elif node.type == "enum_declaration":
                    self._process_enum_node(node, leaf_package)
            # Flush any remaining pending nodes and relationships
            self._flush_batch()
            # Note: The call to await self._batch_explain_code_snippets() has been removed from here.
            # It will be called from main.py after all files are processed.

        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            # Flush batch in case of error to prevent data loss
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

# Example of how to run the analyzer and then the batch explanation
# async def main_analysis_loop(analyzer, java_files_list):
#     for java_file in tqdm(java_files_list, desc="Processing Java files"):
#         analyzer.process_java_file(java_file)
#     # After all files are processed, explain all collected snippets
#     await analyzer._batch_explain_code_snippets()
#     # Final flush to save any updated nodes from batch explanations
#     analyzer._flush_batch() 
#     # Merge duplicate packages if necessary
#     analyzer.merge_duplicate_packages()


def find_java_files(directory: str) -> List[str]:
    """Find all Java files in directory"""
    java_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.java'):
                java_files.append(os.path.join(root, file))
    return java_files 