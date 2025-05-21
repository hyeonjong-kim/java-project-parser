from typing import Generator, Optional, Tuple
from tree_sitter import Language, Parser, Node
import tree_sitter_java as tsjava

class JavaParser:
    """Java source code parser using tree-sitter"""
    
    def __init__(self):
        self.language = Language(tsjava.language())
        self.parser = Parser(self.language)
    
    def parse(self, source_code: str) -> Node:
        """Parse Java source code into AST"""
        return self.parser.parse(bytes(source_code, 'utf8')).root_node

def walk(node: Node) -> Generator[Node, None, None]:
    """Walk through all nodes in the AST"""
    yield node
    for child in node.children:
        yield from walk(child)

def find_node_by_type(node: Node, node_type: str) -> Optional[Node]:
    """Find first node of specific type in AST"""
    return next((n for n in walk(node) if n.type == node_type), None)

def find_node_by_type_in_children(node: Node, node_type: str) -> Optional[Node]:
    """Find first node of specific type in children of AST"""
    return next((n for n in node.children if n.type == node_type), None)

def extract_text(node: Node) -> str:
    """Extract text from node in source code"""
    return node.text.decode('utf-8')

def parse_package_declaration(root: Node) -> Optional[str]:
    """Extract package declaration from AST"""
    pkg_node = find_node_by_type(root, "package_declaration")
    if pkg_node:
        return extract_text(pkg_node).split()[1].rstrip(";")
    return None 