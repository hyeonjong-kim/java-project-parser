from typing import Optional
from py2neo.ogm import GraphObject, Property, RelatedTo, RelatedFrom

class JavaProject(GraphObject):
    """Java project root node that contains all packages"""
    __primarykey__ = "name"
    
    name = Property()
    description = Property()
    summary = Property()
    packages = RelatedFrom("JavaPackage", "PACKAGE")

    def __init__(self, name: str, description: str = "", summary: str = ""):
        self.name = name
        self.description = description
        self.summary = summary

class JavaPackage(GraphObject):
    """Java package that can contain classes and other packages"""
    __primarykey__ = "name"
    
    name = Property()
    description = Property()
    summary = Property()
    is_internal = Property()  # True if package is defined within the project, False for external libraries
    packages = RelatedFrom("JavaPackage", "CHILD_PACKAGE")
    package_by = RelatedTo("JavaPackage", "PARENT_PACKAGE")
    classes = RelatedFrom("JavaClass", "CLASS")
    interfaces = RelatedFrom("JavaInterface", "INTERFACE")
    enums = RelatedFrom("JavaEnum", "ENUM")
    imports = RelatedFrom("JavaPackage", "IMPORT")
    imported_by = RelatedTo("JavaPackage", "IMPORT")
    contained_in_project = RelatedTo("JavaProject", "PACKAGE")

    def __init__(self, name: str, description: str = "", summary: str = "", is_internal: bool = False):
        self.name = name
        self.description = description
        self.summary = summary
        self.is_internal = is_internal

class JavaInterface(GraphObject):
    """Java interface representation"""
    __primarykey__ = "name"
    
    name = Property()
    body = Property()
    description = Property()
    summary = Property()
    extends = RelatedFrom("JavaInterface", "EXTEND")
    methods = RelatedFrom("JavaMethod", "METHOD")
    contained_in = RelatedTo("JavaPackage", "INTERFACE")

    def __init__(self, name: str, body: str, description: str, summary: str):
        self.name = name
        self.body = body
        self.description = description
        self.summary = summary

class JavaClass(GraphObject):
    """Java class representation"""
    __primarykey__ = "name"
    
    name = Property()
    body = Property()
    description = Property()
    summary = Property()
    extends = RelatedFrom("JavaClass", "EXTEND")
    implements = RelatedTo("JavaInterface", "IMPLEMENT")
    methods = RelatedFrom("JavaMethod", "METHOD")
    contained_in = RelatedTo("JavaPackage", "CLASS")

    def __init__(self, name: str, body: str, description: str, summary: str):
        self.name = name
        self.body = body
        self.description = description
        self.summary = summary

class JavaEnum(GraphObject):
    """Java enum representation"""
    __primarykey__ = "name"

    name = Property()   
    body = Property()
    description = Property()
    summary = Property()
    contained_in = RelatedTo("JavaPackage", "ENUM")

    def __init__(self, name: str, body: str, description: str, summary: str):
        self.name = name
        self.body = body
        self.description = description
        self.summary = summary

class JavaMethod(GraphObject):
    """Java method representation"""
    __primarykey__ = "signature"
    
    name = Property()
    signature = Property()  # Added: method_name + parameter types for uniqueness
    body = Property()
    description = Property()
    summary = Property()
    return_type = Property()
    contained_in_class = RelatedTo("JavaClass", "METHOD")
    contained_in_interface = RelatedTo("JavaInterface", "METHOD")

    def __init__(self, name: str, signature: str, body: str, description: str, summary: str, return_type: str):
        self.name = name
        self.signature = signature
        self.body = body
        self.description = description
        self.summary = summary
        self.return_type = return_type 