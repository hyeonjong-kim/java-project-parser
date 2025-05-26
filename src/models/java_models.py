from typing import Optional
from py2neo.ogm import GraphObject, Property, RelatedTo, RelatedFrom

class JavaPackage(GraphObject):
    """Base class for Java packages"""
    __primarykey__ = "name"
    
    name = Property()
    imports = RelatedFrom("JavaPackage", "IMPORT")
    imported_by = RelatedTo("JavaPackage", "IMPORT")

    def __init__(self, name: str):
        self.name = name

class JavaInternalPackage(JavaPackage):
    """Internal package that can contain other packages"""
    __primarykey__ = "name"
    
    name = Property()
    description = Property()
    summary = Property()
    packages = RelatedFrom("JavaPackage", "CHILD_PACKAGE")
    package_by = RelatedTo("JavaPackage", "PARENT_PACKAGE")
    classes = RelatedFrom("JavaClass", "CLASS")
    interfaces = RelatedFrom("JavaInterface", "INTERFACE")
    enums = RelatedFrom("JavaEnum", "ENUM")

    def __init__(self, name: str, description: str = "", summary: str = ""):
        super().__init__(name)
        self.description = description
        self.summary = summary

class JavaLeafPackage(JavaPackage):
    """Leaf package that can contain classes"""
    __primarykey__ = "name"
    
    name = Property()
    description = Property()
    summary = Property()
    classes = RelatedFrom("JavaClass", "CLASS")
    interfaces = RelatedFrom("JavaInterface", "INTERFACE")
    enums = RelatedFrom("JavaEnum", "ENUM")
    package_by = RelatedTo("JavaPackage", "PARENT_PACKAGE")
    def __init__(self, name: str, description: str = "", summary: str = ""):
        super().__init__(name)
        self.description = description
        self.summary = summary

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
    fields = RelatedFrom("JavaField", "FIELD")
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
    constants = RelatedFrom("JavaEnumConstant", "CONSTANT")
    contained_in = RelatedTo("JavaPackage", "ENUM")

    def __init__(self, name: str, body: str, description: str, summary: str):
        self.name = name
        self.body = body
        self.description = description
        self.summary = summary

class JavaEnumConstant(GraphObject):
    """Java enum constant representation"""
    __primarykey__ = "constant"

    constant = Property()
    contained_in = RelatedTo("JavaEnum", "CONSTANT")

    def __init__(self, constant: str):
        self.constant = constant
        

class JavaField(GraphObject):
    """Java field representation"""
    __primarykey__ = "name"
    
    name = Property()
    type = Property()
    contained_in = RelatedTo("JavaClass", "FIELD")

    def __init__(self, name: str, type: str):
        self.name = name
        self.type = type

class JavaMethod(GraphObject):
    """Java method representation"""
    __primarykey__ = "signature"
    
    name = Property()
    signature = Property()  # Added: method_name + parameter types for uniqueness
    body = Property()
    description = Property()
    summary = Property()
    return_type = Property()
    parameters = RelatedFrom("JavaParameter", "PARAMETER")
    local_variables = RelatedFrom("JavaLocalVariable", "VARIABLE")
    contained_in_class = RelatedTo("JavaClass", "METHOD")
    contained_in_interface = RelatedTo("JavaInterface", "METHOD")

    def __init__(self, name: str, signature: str, body: str, description: str, summary: str, return_type: str):
        self.name = name
        self.signature = signature
        self.body = body
        self.description = description
        self.summary = summary
        self.return_type = return_type

class JavaParameter(GraphObject):
    """Java parameter representation"""
    __primarykey__ = "name"
    
    name = Property()
    type = Property()
    contained_in = RelatedTo("JavaMethod", "PARAMETER")

    def __init__(self, name: str, type: str):
        self.name = name
        self.type = type

class JavaLocalVariable(GraphObject):
    """Java local variable representation"""
    __primarykey__ = "name"
    
    name = Property()
    type = Property()
    contained_in = RelatedTo("JavaMethod", "VARIABLE")

    def __init__(self, name: str, type: str):
        self.name = name
        self.type = type 