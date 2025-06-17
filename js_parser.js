const acorn = require('acorn');
const walk = require('acorn-walk');

class JavaScriptCodeVisitor {
    /**
     * JavaScript AST visitor to extract code elements with enhanced metadata
     */

    constructor(filePath, is_dependency = false) {
        this.filePath = filePath;
        this.is_dependency = is_dependency;
        this.functions = [];
        this.classes = [];
        this.variables = [];
        this.imports = [];
        this.exports = [];
        this.functionCalls = [];
        this.currentContext = null;
        this.currentClass = null;
        this.sourceCode = '';
        this.sourceLines = [];
        this.is_exports = false;
        this.comments = []; // Store all comments with positions
        this.processedNodes = new Set();
    }

    /**
     * Parse JavaScript code and extract all elements
     */
    parse(sourceCode) {
        this.sourceCode = sourceCode;
        this.sourceLines = sourceCode.split(/\r?\n/);

        try {
            const ast = acorn.parse(sourceCode, {
                ecmaVersion: 2022,
                sourceType: 'module',
                locations: true,
                ranges: true,
                allowReturnOutsideFunction: true,
                onComment: (isBlock, text, start, end, startLoc, endLoc) => {
                    this.comments.push({
                        type: isBlock ? 'Block' : 'Line',
                        value: text,
                        start,
                        end,
                        loc: { start: startLoc, end: endLoc },
                        used: false,
                        isJSDoc: isBlock && text.trim().startsWith('*')
                    });
                }
            });


            this.visitNode(ast);
            return this.getResults();
        } catch (error) {
            console.log(error)
            return this.getResults();
        }
    }

    /**
     * Visit AST nodes recursively
     */
    visitNode(node) {
        if (!node || !node.type) return;

        const nodeId = `${node.type}:${node.range ? node.range[0] : ''}:${node.range ? node.range[1] : ''}`;
        if (this.processedNodes.has(nodeId)) {
            return;
        }

        switch (node.type) {
            case 'FunctionDeclaration':
                this.visitFunctionDeclaration(node);
                break;
            case 'FunctionExpression':
                //     if (!this.isPartOfMethodDefinition(node) && !this.isPartOfVariableDeclaration(node)) {
                //         this.visitFunctionExpression(node);
                //     }
                break;
            case 'ArrowFunctionExpression':
                //     if (!this.isPartOfMethodDefinition(node) && !this.isPartOfVariableDeclaration(node)) {
                //         this.visitFunctionExpression(node);
                //     }
                break;
            case 'ClassDeclaration':
                this.visitClassDeclaration(node);
                break;
            case 'MethodDefinition':
                this.visitMethodDefinition(node);
                break;
            case 'VariableDeclaration':
                this.visitVariableDeclaration(node);
                break;
            case 'ImportDeclaration':
                this.visitImportDeclaration(node);
                break;
            case 'ExportNamedDeclaration':
            case 'ExportDefaultDeclaration':
            case 'ExportAllDeclaration':
                this.visitExportDeclaration(node);
                break;
            case 'CallExpression':
                this.visitCallExpression(node);
                break;
            case 'AssignmentExpression':
                this.visitAssignmentExpression(node);
                break;
        }

        this.processedNodes.add(nodeId);
        this.visitChildren(node);
    }

    /**
     * Enhanced docstring extraction that handles multiple comment formats
     */
    getDocstring(node) {
        if (!node.loc) return null;

        const nodeStartLine = node.loc.start.line;
        const nodeStartColumn = node.loc.start.column;

        // Find the closest preceding comment block
        let closestComment = null;
        let minDistance = Infinity;

        for (const comment of this.comments) {
            if (comment.used || comment.type !== 'JSDoc') continue;
            const commentEndLine = comment.loc.end.line;

            // Only consider comments that end before the node starts
            if (commentEndLine < nodeStartLine ||
                (commentEndLine === nodeStartLine && comment.loc.end.column < nodeStartColumn)) {

                const distance = nodeStartLine - commentEndLine;

                // JSDoc comments get priority, but we also consider regular block comments
                const isRelevant = comment.type === 'Block' && (
                    comment.isJSDoc ||
                    comment.value.includes('@param') ||
                    comment.value.includes('@return') ||
                    comment.value.includes('@description') ||
                    distance <= 2 // Close proximity for non-JSDoc blocks
                );

                if (isRelevant && distance < minDistance) {
                    minDistance = distance;
                    closestComment = comment;
                }
            }
        }

        if (closestComment) {
            closestComment.used = true;
            return this.parseDocstring(closestComment);
        }

        // Fallback: look for inline comments using line-by-line approach
        return this.getInlineDocstring(nodeStartLine);
    }

    /**
     * Parse structured docstring from comment
     */
    parseDocstring(comment) {
        const lines = comment.value.split('\n').map(line =>
            line.trim().replace(/^\*+\s?/, '').trim()
        ).filter(line => line.length > 0);

        if (lines.length === 0) return null;

        const result = {
            description: '',
            params: [],
            returns: null,
            example: null,
            raw: comment.value
        };

        let currentSection = 'description';
        let exampleLines = [];

        for (let line of lines) {
            line = line.trim();

            if (line.startsWith('@param')) {
                currentSection = 'params';
                const paramMatch = line.match(/@param\s*\{([^}]+)\}\s*(\w+)\s*(.*)/);
                if (paramMatch) {
                    result.params.push({
                        name: paramMatch[2],
                        type: paramMatch[1],
                        description: paramMatch[3] || ''
                    });
                } else {
                    // Handle @param without type
                    const simpleParamMatch = line.match(/@param\s+(\w+)\s*(.*)/);
                    if (simpleParamMatch) {
                        result.params.push({
                            name: simpleParamMatch[1],
                            type: 'any',
                            description: simpleParamMatch[2] || ''
                        });
                    }
                }
            } else if (line.startsWith('@returns') || line.startsWith('@return')) {
                currentSection = 'returns';
                const returnMatch = line.match(/@returns?\s*\{([^}]+)\}\s*(.*)/);
                if (returnMatch) {
                    result.returns = {
                        type: returnMatch[1],
                        description: returnMatch[2] || ''
                    };
                } else {
                    // Handle @returns without type
                    const simpleReturnMatch = line.match(/@returns?\s+(.*)/);
                    if (simpleReturnMatch) {
                        result.returns = {
                            type: 'any',
                            description: simpleReturnMatch[1] || ''
                        };
                    }
                }
            } else if (line.startsWith('@example')) {
                currentSection = 'example';
                exampleLines = [];
            } else if (line.startsWith('@')) {
                // Other JSDoc tags - could extend this
                currentSection = 'other';
            } else {
                // Regular content
                if (currentSection === 'description') {
                    result.description += (result.description ? ' ' : '') + line;
                } else if (currentSection === 'example') {
                    exampleLines.push(line);
                } else if (currentSection === 'params' && result.params.length > 0) {
                    // Continue description of last parameter
                    const lastParam = result.params[result.params.length - 1];
                    lastParam.description += (lastParam.description ? ' ' : '') + line;
                } else if (currentSection === 'returns' && result.returns) {
                    // Continue description of return value
                    result.returns.description += (result.returns.description ? ' ' : '') + line;
                }
            }
        }

        if (exampleLines.length > 0) {
            result.example = exampleLines.join('\n');
        }

        // Clean up descriptions
        result.description = result.description.trim();
        if (result.returns) {
            result.returns.description = result.returns.description.trim();
        }
        result.params.forEach(param => {
            param.description = param.description.trim();
        });

        return result;
    }

    /**
     * Fallback method for extracting docstrings from source lines
     */
    getInlineDocstring(nodeStartLine) {
        let docLines = [];
        let foundDocBlock = false;

        // Look backwards from the node line
        for (let i = nodeStartLine - 2; i >= 0; i--) {
            const line = this.sourceLines[i];
            if (!line) break;

            const trimmed = line.trim();

            if (trimmed.endsWith('*/')) {
                foundDocBlock = true;
                const content = trimmed.replace(/\*\/$/, '').replace(/^\s*\*\s?/, '');
                if (content) docLines.unshift(content);
            } else if (foundDocBlock && (trimmed.startsWith('*') || trimmed.startsWith('/**'))) {
                const content = trimmed.replace(/^\/?\*+\s?/, '');
                if (content) docLines.unshift(content);
                if (trimmed.startsWith('/**')) break;
            } else if (foundDocBlock) {
                break;
            } else if (trimmed === '' || trimmed.startsWith('//')) {
                // Skip empty lines and single-line comments
                continue;
            } else {
                break;
            }
        }

        if (docLines.length > 0) {
            return this.parseDocstring({ value: docLines.join('\n') });
        }

        return null;
    }

    /**
     * Get return type from JSDoc comments or inferred from code
     */
    getReturnType(node) {
        const docstring = this.getDocstring(node);
        if (docstring && docstring.returns) {
            return docstring.returns.type;
        }

        // Try to infer from function body
        if (node.body && node.body.type === 'BlockStatement') {
            const bodySource = this.getSourceCode(node.body);
            if (bodySource.includes('return ')) {
                // Simple inference - could be made more sophisticated
                if (bodySource.includes('return true') || bodySource.includes('return false')) {
                    return 'boolean';
                } else if (bodySource.match(/return\s+['"`]/)) {
                    return 'string';
                } else if (bodySource.match(/return\s+\d+/)) {
                    return 'number';
                } else if (bodySource.includes('return []')) {
                    return 'array';
                } else if (bodySource.includes('return {}')) {
                    return 'object';
                }
                return 'any';
            }
        }

        return null;
    }

    /**
     * Visit function declarations with enhanced docstring extraction
     */
    visitFunctionDeclaration(node) {
        const docstring = this.getDocstring(node);
        const funcData = {
            name: node.id ? node.id.name : '<anonymous>',
            type: 'function',
            line_number: node.loc ? node.loc.start.line : null,
            end_line: node.loc ? node.loc.end.line : null,
            startColumn: node.loc ? node.loc.start.column : null,
            endColumn: node.loc ? node.loc.end.column : null,
            args: node.params.map(param => this.getParameterName(param)),
            source: this.getSourceCode(node),
            context: this.currentContext,
            class_context: this.currentClass,
            is_dependency: this.is_dependency,
            isAsync: node.async,
            isGenerator: node.generator,
            docstring: JSON.stringify(docstring),
            is_Export: this.is_exports
        };

        const prevContext = this.currentContext;
        this.currentContext = funcData.name;
        this.functions.push(funcData);

        this.visitNode(node.body);
        this.currentContext = prevContext;
    }

    /**
     * Visit function expressions with enhanced docstring extraction
     */
    visitFunctionExpression(node) {
        const name = node.id ? node.id.name : this.inferFunctionName(node);
        const docstring = this.getDocstring(node);

        if (!name) return

        const funcData = {
            name: name,
            type: node.type === 'ArrowFunctionExpression' ? 'arrow_function' : 'function_expression',
            line_number: node.loc ? node.loc.start.line : null,
            end_line: node.loc ? node.loc.end.line : null,
            startColumn: node.loc ? node.loc.start.column : null,
            endColumn: node.loc ? node.loc.end.column : null,
            args: node.params.map(param => this.getParameterName(param)),
            source: this.getSourceCode(node),
            context: this.currentContext,
            class_context: this.currentClass,
            is_dependency: this.is_dependency,
            isAsync: node.async,
            isGenerator: node.generator,
            docstring: JSON.stringify(docstring),
            is_Export: this.is_exports
        };

        const prevContext = this.currentContext;
        this.currentContext = funcData.name;
        this.functions.push(funcData);

        this.visitNode(node.body);
        this.currentContext = prevContext;
    }

    visitObjectExpression(node) {
        const prevContext = this.currentContext;
        this.currentContext = node.left.name;

        for (const prop of node.right.properties) {
            if (prop.type === 'Property') {
                const value = prop.value;
                if (value.type === 'FunctionExpression' || value.type === 'ArrowFunctionExpression') {
                    value.id = { name: prop.key.name };
                    this.visitFunctionExpression(value);
                } else if (value.type === 'NewExpression') {
                    const newExprData = {
                        name: `${node.left.name}.${prop.key.name}`,
                        type: 'object_property_new_expression',
                        line_number: prop.loc ? prop.loc.start.line : null,
                        end_line: prop.loc ? prop.loc.end.line : null,
                        startColumn: prop.loc ? prop.loc.start.column : null,
                        endColumn: prop.loc ? prop.loc.end.column : null,
                        constructor: value.callee.name || this.getSourceCode(value.callee),
                        arguments: value.arguments.map(arg => this.getSourceCode(arg)),
                        value: this.getSourceCode(value),
                        context: this.currentContext,
                        class_context: this.currentClass,
                        is_dependency: this.is_dependency,
                        kind : null
                    };
                    this.variables.push(newExprData);
                }
                else {
                    this.visitNode(value);
                }
            }
        }

        this.currentContext = prevContext;
    }

    /**
     * Visit method definitions with enhanced docstring extraction
     */
    visitMethodDefinition(node) {
        const docstring = this.getDocstring(node);

        const methodData = {
            name: this.getPropertyName(node.key),
            type: 'method',
            line_number: node.loc ? node.loc.start.line : null,
            end_line: node.loc ? node.loc.end.line : null,
            startColumn: node.loc ? node.loc.start.column : null,
            endColumn: node.loc ? node.loc.end.column : null,
            args: node.value.params.map(param => this.getParameterName(param)),
            source: this.getSourceCode(node),
            context: this.currentContext,
            class_context: this.currentClass,
            is_dependency: this.is_dependency,
            kind: node.kind,
            isStatic: node.static,
            isAsync: node.value.async,
            isGenerator: node.value.generator,
            docstring: JSON.stringify(docstring),
            is_Export: this.is_exports
        };

        const prevContext = this.currentContext;
        this.currentContext = methodData.name;
        this.functions.push(methodData);

        this.visitNode(node.value.body);
        this.currentContext = prevContext;
    }

    /**
     * Get comments that are associated with a specific node
     */
    getNodeComments(node) {
        if (!node.loc) return [];

        const nodeStartLine = node.loc.start.line;
        const associatedComments = [];

        // Find comments within 3 lines before the node
        for (const comment of this.comments) {
            const commentEndLine = comment.loc.end.line;
            const distance = nodeStartLine - commentEndLine;

            if (distance >= 0 && distance <= 3) {
                associatedComments.push({
                    type: comment.type,
                    value: comment.value.trim(),
                    line: comment.loc.start.line,
                    isJSDoc: comment.isJSDoc
                });
            }
        }

        return associatedComments;
    }

    // ... (keep all other existing methods unchanged) ...

    isPartOfMethodDefinition(node) {
        let parent = node.parent;
        while (parent) {
            if (parent.type === 'MethodDefinition') {
                return parent.value === node;
            }
            parent = parent.parent;
        }
        return false;
    }

    visitClassDeclaration(node) {
        const docstring = this.getDocstring(node);

        const classData = {
            name: node.id ? node.id.name : '<anonymous>',
            type: 'class',
            line_number: node.loc ? node.loc.start.line : null,
            end_line: node.loc ? node.loc.end.line : null,
            startColumn: node.loc ? node.loc.start.column : null,
            endColumn: node.loc ? node.loc.end.column : null,
            bases: node.superClass ? this.getIdentifierName(node.superClass) : null,
            source: this.getSourceCode(node),
            context: this.currentContext,
            is_dependency: this.is_dependency,
            is_Export: this.is_exports
        };

        const prevContext = this.currentContext;
        const prevClass = this.currentClass;
        this.currentContext = classData.name;
        this.currentClass = classData.name;
        this.classes.push(classData);

        this.visitNode(node.body);

        this.currentContext = prevContext;
        this.currentClass = prevClass;
    }

    visitVariableDeclaration(node) {
        node.declarations.forEach(declaration => {

            if (declaration.id && declaration.id.type === 'Identifier') {
                const docstring = this.getDocstring(node);

                if (declaration.init && declaration.init.type === 'ObjectExpression') {
                    for (const prop of declaration.init.properties) {
                        if (prop.type === 'Property') {
                            const value = prop.value;
                            if (value.type === 'FunctionExpression') {
                                value.id = {};
                                value.id.name = prop.key.name
                                const prevContext = this.currentContext;
                                this.currentContext = declaration.id.name;
                                this.visitFunctionExpression(value);
                                this.currentContext = prevContext;
                            }
                            else if (value.type === 'ArrowFunctionExpression') {
                                value.id = {};
                                value.id.name = prop.key.name
                                const prevContext = this.currentContext;
                                this.currentContext = declaration.id.name;
                                this.visitFunctionExpression(value); // Or a separate arrow handler
                                this.currentContext = prevContext;
                            }
                            else {
                                value.id = {};
                                value.id.name = prop.key.name
                                const prevContext = this.currentContext;
                                this.currentContext = declaration.id.name;
                                this.visitNode(value); // Or a separate arrow handler
                                this.currentContext = prevContext;
                            }
                        }
                    }
                }

                if (declaration.init && declaration.init.type === "FunctionExpression") {
                    declaration.init.id = {}
                    declaration.init.id.name = declaration.id.name
                    this.visitFunctionExpression(declaration.init);
                    return
                }
                if (declaration.init && declaration.init.type === "ArrowFunctionExpression") {
                    declaration.init.id = {}
                    declaration.init.id.name = declaration.id.name
                    // declaration.init.id = declaration.id.name
                    this.visitFunctionExpression(declaration.init);
                    return
                }
                if (declaration.init && declaration.init.type === "ClassExpression") {
                    declaration.init.id = {}
                    declaration.init.id.name = declaration.id.name
                    // declaration.init.id = declaration.id.name
                    this.visitClassDeclaration(declaration.init);
                    return
                }

                const varData = {
                    name: declaration.id.name,
                    type: 'variable',
                    kind: null,
                    line_number: node.loc ? node.loc.start.line : null,
                    startColumn: node.loc ? node.loc.start.column : null,
                    endColumn: node.loc ? node.loc.end.column : null,
                    value: declaration.init ? this.getSourceCode(declaration.init) : null,
                    context: this.currentContext,
                    class_context: this.currentClass,
                    is_dependency: this.is_dependency
                };
                this.variables.push(varData);

                // Handle require() calls
                if (declaration.init && declaration.init.type === 'CallExpression' &&
                    declaration.init.callee.type === 'Identifier' &&
                    declaration.init.callee.name === 'require') {

                    const requireArg = declaration.init.arguments[0];
                    if (requireArg && requireArg.type === 'Literal') {
                        const importData = {
                            name: declaration.id.name,
                            type: 'require_import',
                            line_number: node.loc ? node.loc.start.line : null,
                            startColumn: node.loc ? node.loc.start.column : null,
                            endColumn: node.loc ? node.loc.end.column : null,
                            context: this.currentContext,
                            is_dependency: this.is_dependency,
                            moduleSource: requireArg.value,
                            alias: null
                        };
                        this.imports.push(importData);
                    }
                }
            }
        });
    }

    visitImportDeclaration(node) {
        const moduleSource = node.source.value;

        node.specifiers.forEach(specifier => {
            let importData = {
                line_number: node.loc ? node.loc.start.line : null,
                startColumn: node.loc ? node.loc.start.column : null,
                endColumn: node.loc ? node.loc.end.column : null,
                context: this.currentContext,
                is_dependency: this.is_dependency,
                moduleSource: moduleSource
            };

            switch (specifier.type) {
                case 'ImportDefaultSpecifier':
                    importData.name = specifier.local.name;
                    importData.type = 'default_import';
                    importData.alias = null;
                    break;
                case 'ImportSpecifier':
                    importData.name = specifier.imported.name;
                    importData.type = 'named_import';
                    importData.alias = specifier.local.name !== specifier.imported.name ? specifier.local.name : null;
                    break;
                case 'ImportNamespaceSpecifier':
                    importData.name = '*';
                    importData.type = 'namespace_import';
                    importData.alias = specifier.local.name;
                    break;
            }

            this.imports.push(importData);
        });
    }

    visitExportDeclaration(node) {
        const exportData = {
            line_number: node.loc ? node.loc.start.line : null,
            startColumn: node.loc ? node.loc.start.column : null,
            endColumn: node.loc ? node.loc.end.column : null,
            context: this.currentContext,
            is_dependency: this.is_dependency,
            source: this.getSourceCode(node)
        };

        switch (node.type) {
            case 'ExportDefaultDeclaration':
                this.is_exports = true;
                this.visitChildren(node);
                this.is_exports = false;
                break;
            case 'ExportNamedDeclaration':
                exportData.type = 'named_export';
                if (node.specifiers.length > 0) {
                    node.specifiers.forEach(specifier => {
                        const namedExport = {
                            ...exportData,
                            name: specifier.exported.name,
                            localName: specifier.local.name
                        };
                        this.exports.push(namedExport);
                    });
                    return;
                } else if (node.declaration) {
                    this.is_exports = true;
                    this.visitChildren(node);
                    this.is_exports = false;
                    return;
                }
                break;
            case 'ExportAllDeclaration':
                exportData.type = 'export_all';
                exportData.name = '*';
                exportData.moduleSource = node.source ? node.source.value : null;
                break;
        }

        this.exports.push(exportData);
    }

    visitCallExpression(node) {
        // Early return for require() calls - handled elsewhere
        if (node.callee.type === 'Identifier' && node.callee.name === 'require') {
            return;
        }

        const callInfo = this.extractCallInfo(node);

        // Skip if call extraction failed
        if (!callInfo.name) {
            return;
        }

        // Apply filters
        if (this.shouldSkipCall(callInfo)) {
            return;
        }

        const callData = {
            name: callInfo.name,
            full_name: callInfo.fullName,
            type: callInfo.type,
            line_number: node.loc ? node.loc.start.line : null,
            end_line: node.loc ? node.loc.end.line : null,
            startColumn: node.loc ? node.loc.start.column : null,
            endColumn: node.loc ? node.loc.end.column : null,
            args: this.extractArguments(node.arguments),
            context: this.currentContext,
            class_context: this.currentClass,
            is_dependency: this.is_dependency,
            ...callInfo.metadata
        };

        this.functionCalls.push(callData);
    }

    // Extract comprehensive call information
    extractCallInfo(node) {
        const callee = node.callee;

        switch (callee.type) {
            case 'Identifier':
                return this.handleIdentifierCall(callee);

            case 'MemberExpression':
                return this.handleMemberExpressionCall(callee);

            case 'CallExpression':
                return this.handleChainedCall(callee);

            case 'FunctionExpression':
            case 'ArrowFunctionExpression':
                return this.handleIIFE(callee);

            case 'ConditionalExpression':
                return this.handleConditionalCall(callee);

            case 'TaggedTemplateExpression':
                return this.handleTaggedTemplate(callee);

            default:
                return {
                    name: this.getCallName(callee),
                    fullName: this.getCallName(callee),
                    type: 'complex_call',
                    metadata: { calleeType: callee.type }
                };
        }
    }

    // Handle direct function calls: func()
    handleIdentifierCall(callee) {
        return {
            name: callee.name,
            fullName: callee.name,
            type: 'direct_call',
            metadata: {}
        };
    }

    // Handle method calls: obj.method(), Class.staticMethod()
    handleMemberExpressionCall(callee) {
        const memberName = this.getMemberExpressionName(callee);
        const methodName = callee.property.name || this.getSourceCode(callee.property);

        // Determine call type based on object
        let callType = 'method_call';
        let objectInfo = {};

        if (callee.object.type === 'Identifier') {
            const objectName = callee.object.name;

            // Check for common patterns
            if (objectName === 'console') {
                callType = 'console_call';
            } else if (objectName === 'Math') {
                callType = 'math_call';
            } else if (objectName === 'JSON') {
                callType = 'json_call';
            } else if (objectName === 'Array') {
                callType = 'array_static_call';
            } else if (objectName === 'Object') {
                callType = 'object_static_call';
            } else if (objectName === 'String') {
                callType = 'string_static_call';
            } else if (objectName === 'Number') {
                callType = 'number_static_call';
            } else if (this.isClassName(objectName)) {
                callType = 'static_method_call';
            } else if (objectName === 'this') {
                callType = 'this_method_call';
            } else if (objectName === 'super') {
                callType = 'super_method_call';
            }

            objectInfo = {
                object: objectName,
                property: methodName,
                computed: callee.computed
            };
        } else {
            // Complex object expression
            objectInfo = {
                object: this.getSourceCode(callee.object),
                property: methodName,
                computed: callee.computed
            };
        }

        return {
            name: methodName,
            fullName: memberName,
            type: callType,
            metadata: objectInfo
        };
    }

    // Handle chained calls: func()()
    handleChainedCall(callee) {
        const chainedCallName = this.getCallName(callee);
        return {
            name: 'chained_call',
            fullName: `${chainedCallName}()`,
            type: 'chained_call',
            metadata: {
                innerCall: chainedCallName
            }
        };
    }

    // Handle IIFE: (function() {})()
    handleIIFE(callee) {
        return {
            name: 'IIFE',
            fullName: 'IIFE',
            type: 'iife',
            metadata: {
                functionType: callee.type,
                isAsync: callee.async || false,
                isGenerator: callee.generator || false
            }
        };
    }

    // Handle conditional calls: (condition ? func1 : func2)()
    handleConditionalCall(callee) {
        return {
            name: 'conditional_call',
            fullName: this.getSourceCode(callee),
            type: 'conditional_call',
            metadata: {
                test: this.getSourceCode(callee.test),
                consequent: this.getSourceCode(callee.consequent),
                alternate: this.getSourceCode(callee.alternate)
            }
        };
    }

    // Handle tagged template literals: tag`template`
    handleTaggedTemplate(callee) {
        return {
            name: 'tagged_template',
            fullName: this.getSourceCode(callee),
            type: 'tagged_template',
            metadata: {}
        };
    }

    // Enhanced argument extraction
    extractArguments(args) {
        return args.map((arg, index) => {
            const argData = this.getSourceCode(arg) ;

            // // Add specific metadata for different argument types
            // switch (arg.type) {
            //     case 'FunctionExpression':
            //     case 'ArrowFunctionExpression':
            //         argData.isCallback = true;
            //         argData.isAsync = arg.async || false;
            //         argData.paramCount = arg.params.length;
            //         break;

            //     case 'ObjectExpression':
            //         argData.propertyCount = arg.properties.length;
            //         break;

            //     case 'ArrayExpression':
            //         argData.elementCount = arg.elements.length;
            //         break;

            //     case 'Literal':
            //         argData.literalType = typeof arg.value;
            //         break;

            //     case 'Identifier':
            //         argData.isVariable = true;
            //         break;
            // }

            return argData;
        });
    }

    // Determine if a call should be skipped
    shouldSkipCall(callInfo) {
        // Skip logger calls
        if (callInfo.fullName.includes('logger.') || callInfo.fullName.includes('console.')) {
            return true;
        }

        // Skip complex calls that couldn't be resolved
        if (callInfo.name === "<complex>") {
            return true;
        }

        // Built-in JavaScript functions and methods to skip
        const builtinFunctions = new Set([
            // Global functions
            'parseInt', 'parseFloat', 'isNaN', 'isFinite', 'encodeURI', 'decodeURI',
            'encodeURIComponent', 'decodeURIComponent', 'escape', 'unescape',
            'eval', 'setTimeout', 'setInterval', 'clearTimeout', 'clearInterval',

            // Array methods
            'push', 'pop', 'shift', 'unshift', 'slice', 'splice', 'join', 'reverse',
            'sort', 'concat', 'indexOf', 'lastIndexOf', 'forEach', 'map', 'filter',
            'reduce', 'reduceRight', 'every', 'some', 'find', 'findIndex', 'includes',

            // String methods
            'charAt', 'charCodeAt', 'substring', 'substr', 'slice', 'toLowerCase',
            'toUpperCase', 'trim', 'replace', 'split', 'match', 'search', 'indexOf',
            'lastIndexOf', 'startsWith', 'endsWith', 'includes', 'repeat', 'padStart', 'padEnd',

            // Object methods
            'toString', 'valueOf', 'hasOwnProperty', 'isPrototypeOf', 'propertyIsEnumerable',

            // Promise methods
            'then', 'catch', 'finally'
        ]);

        // Skip if it's a built-in function
        if (builtinFunctions.has(callInfo.name)) {
            return true;
        }

        // Skip console methods specifically
        if (callInfo.type === 'console_call' || callInfo.type === 'iife') {
            return true;
        }

        // Skip certain built-in static calls unless explicitly tracking them
        if (['math_call', 'json_call'].includes(callInfo.type) && !this.trackBuiltinCalls) {
            return true;
        }

        return false;
    }

    // Helper method to check if a name is likely a class name (PascalCase)
    isClassName(name) {
        return /^[A-Z][a-zA-Z0-9]*$/.test(name);
    }

    // Enhanced getMemberExpressionName with better handling
    getMemberExpressionName(memberExpr) {
        if (memberExpr.computed) {
            const objectName = memberExpr.object.type === 'Identifier' ?
                memberExpr.object.name : this.getSourceCode(memberExpr.object);
            const propertyName = this.getSourceCode(memberExpr.property);
            return `${objectName}[${propertyName}]`;
        }

        const objectName = memberExpr.object.type === 'Identifier' ?
            memberExpr.object.name : this.getSourceCode(memberExpr.object);
        const propertyName = memberExpr.property.type === 'Identifier' ?
            memberExpr.property.name : this.getSourceCode(memberExpr.property);

        return `${objectName}.${propertyName}`;
    }

    // Helper method to create base location data
    getLocationData(node) {
        return node.loc ? {
            line_number: node.loc.start.line,
            end_line: node.loc.end.line,
            startColumn: node.loc.start.column,
            endColumn: node.loc.end.column
        } : {
            line_number: null,
            end_line: null,
            startColumn: null,
            endColumn: null
        };
    }

    // Helper method to create base context data
    getContextData() {
        return {
            context: this.currentContext,
            class_context: this.currentClass,
            is_dependency: this.is_dependency
        };
    }

    visitAssignmentExpression(node) {


        const { left, right } = node;
        const isFunctionRight = right.type === 'FunctionExpression' || right.type === 'ArrowFunctionExpression';

        // Handle function assignments

        if (isFunctionRight && (left.type === 'Identifier' || left.type === 'MemberExpression')) {
            this.handleFunctionAssignment(node, left, right);
            return;
        }

        // Handle NewExpression assignments
        if (right.type === 'NewExpression' && left.type === 'Identifier') {
            this.handleNewExpressionAssignment(node, left, right);
            return;
        }

        // Handle class assignments
        if (right.type === 'ClassExpression' && left.type === 'Identifier') {
            right.id = { name: left.name };
            this.visitClassDeclaration(right);
            return;
        }

        // Handle object assignments
        if (right.type === 'ObjectExpression' && left.type === 'Identifier') {
            this.visitObjectExpression(node);
            return;
        }

        // Handle other assignments to Identifier
        if (left.type === 'Identifier') {
            this.handleGenericAssignment(node, left, right);
        }

        // Handle MemberExpression assignments (for non-function cases)
        else if (left.type === 'MemberExpression') {
            this.handleMemberExpressionAssignment(node, left, right);
        }
    }

    // Handle function assignments (both Identifier and MemberExpression)
    handleFunctionAssignment(node, left, right) {
        const docstring = this.getDocstring(node);
        const isIdentifier = left.type === 'Identifier';
        const name = isIdentifier ? left.name : this.getMemberExpressionName(left);

        const funcData = {
            name,
            type: isIdentifier ? 'assigned_function' : 'member_assigned_function',
            ...this.getLocationData(node),
            args: right.params.map(param => this.getParameterName(param)),
            source: this.getSourceCode(right),
            ...this.getContextData(),
            isAsync: right.async,
            isGenerator: right.generator,
            docstring,
            is_exports: this.is_exports,
            returnType: this.getReturnType(right)
        };

        // Add member-specific properties
        if (!isIdentifier) {
            Object.assign(funcData, {
                object: left.object.name || this.getSourceCode(left.object),
                property: left.property.name || this.getSourceCode(left.property),
                computed: left.computed
            });
        }

        this.functions.push(funcData);
    }

    // Handle NewExpression assignments
    handleNewExpressionAssignment(node, left, right) {
        const newExprData = {
            name: left.name,
            type: 'new_expression_assignment',
            ...this.getLocationData(node),
            constructor: right.callee.name || this.getSourceCode(right.callee),
            arguments: right.arguments.map(arg => this.getSourceCode(arg)),
            value: this.getSourceCode(right),
            kind: null,
            ...this.getContextData()
        };

        this.variables.push(newExprData);
    }

    // Handle generic assignments to Identifier
    handleGenericAssignment(node, left, right) {
        const assignmentData = {
            name: left.name,
            type: 'assignment',
            ...this.getLocationData(node),
            value: this.getSourceCode(right),
            rightType: right.type,
            kind:null , 
            ...this.getContextData()
        };

        this.variables.push(assignmentData);

        // Continue visiting the right side
        this.visitNode(right);
    }

    // Handle MemberExpression assignments (non-function cases)
    handleMemberExpressionAssignment(node, left, right) {
        const memberName = this.getMemberExpressionName(left);

        // Handle NewExpression to MemberExpression
        if (right.type === 'NewExpression') {
            const newExprData = {
                name: memberName,
                kind : null,
                type: 'member_new_expression_assignment',
                ...this.getLocationData(node),
                constructor: right.callee.name || this.getSourceCode(right.callee),
                arguments: right.arguments.map(arg => this.getSourceCode(arg)),
                value: this.getSourceCode(right),
                ...this.getContextData(),
                object: left.object.name || this.getSourceCode(left.object),
                property: left.property.name || this.getSourceCode(left.property),
                computed: left.computed
            };

            this.variables.push(newExprData);
            return;
        }

        // Handle ClassExpression to MemberExpression
        if (right.type === 'ClassExpression') {
            right.id = { name: memberName };
            this.visitClassDeclaration(right);
            return;
        }

        // Handle ObjectExpression to MemberExpression
        if (right.type === 'ObjectExpression') {
            const prevContext = this.currentContext;
            this.currentContext = memberName;

            this.visitObjectExpressionProperties(right.properties);

            this.currentContext = prevContext;
            return;
        }

        // Handle other MemberExpression assignments
        const assignmentData = {
            name: memberName,
            type: 'member_assignment',
            leftType: null,
            rightType: right.type,
            ...this.getLocationData(node),
            left: this.getSourceCode(left),
            value: this.getSourceCode(right),
            ...this.getContextData(),
            object: left.object.name || this.getSourceCode(left.object),
            property: left.property.name || this.getSourceCode(left.property),
            computed: left.computed,
            kind: null
        };

        this.variables.push(assignmentData);

        // Continue visiting the right side
        this.visitNode(right);
    }

    // Helper method to visit object expression properties
    visitObjectExpressionProperties(properties) {
        for (const prop of properties) {
            if (prop.type === 'Property') {
                const value = prop.value;

                if (value.type === 'FunctionExpression' || value.type === 'ArrowFunctionExpression') {
                    value.id = { name: prop.key.name };
                    this.visitFunctionExpression(value);
                } else if (value.type === 'NewExpression') {
                    const newExprData = {
                        name: `${this.currentContext}.${prop.key.name}`,
                        type: 'object_property_new_expression',
                        ...this.getLocationData(prop),
                        constructor: value.callee.name || this.getSourceCode(value.callee),
                        arguments: value.arguments.map(arg => this.getSourceCode(arg)),
                        source: this.getSourceCode(value),
                        ...this.getContextData()
                    };

                    this.newExpressions = this.newExpressions || [];
                    this.newExpressions.push(newExprData);
                } else {
                    this.visitNode(value);
                }
            }
        }
    }

    visitChildren(node) {
        for (const key in node) {
            if (key === 'start' || key === 'end' || key === 'loc' || key === 'range' || key == 'type') continue;

            const child = node[key];


            if (Array.isArray(child)) {
                child.forEach(item => {
                    if (item && typeof item === 'object' && item.type) {
                        this.visitNode(item);
                    }
                });
            } else if (child && typeof child === 'object' && child.type) {
                this.visitNode(child);
            }
        }
    }

    // Helper methods remain the same
    getParameterName(param) {
        switch (param.type) {
            case 'Identifier':
                return param.name;
            case 'RestElement':
                return `...${param.argument.name}`;
            case 'AssignmentPattern':
                return `${param.left.name} = ${this.getSourceCode(param.right)}`;
            case 'ObjectPattern':
                return `{${param.properties.map(p => this.getPropertyName(p.key)).join(', ')}}`;
            case 'ArrayPattern':
                return `[${param.elements.map(e => e ? this.getParameterName(e) : '').join(', ')}]`;
            default:
                return param.name || '<complex>';
        }
    }

    getIdentifierName(node) {
        if (node.type === 'Identifier') {
            return node.name;
        } else if (node.type === 'MemberExpression') {
            return `${this.getIdentifierName(node.object)}.${this.getIdentifierName(node.property)}`;
        }
        return '<complex>';
    }

    getPropertyName(node) {
        if (node.type === 'Identifier') {
            return node.name;
        } else if (node.type === 'Literal') {
            return String(node.value);
        }
        return '<computed>';
    }

    getCallName(node) {
        if (node.type === 'Identifier') {
            return node.name;
        } else if (node.type === 'MemberExpression') {
            return `${this.getIdentifierName(node.object)}.${this.getIdentifierName(node.property)}`;
        }
        return '<complex>';
    }

    getDeclarationName(node) {
        switch (node.type) {
            case 'FunctionDeclaration':
            case 'ClassDeclaration':
                return node.id ? node.id.name : '<anonymous>';
            case 'VariableDeclaration':
                return node.declarations.map(d => d.id.name).join(', ');
            default:
                return '<complex>';
        }
    }

    inferFunctionName(node) {
        if (this.currentContext) {
            return `<anonymous in ${this.currentContext}>`;
        }
        return '<anonymous>';
    }

    getSourceCode(node) {
        if (node.range && this.sourceCode) {
            return this.sourceCode.slice(node.range[0], node.range[1]);
        }
        return '';
    }

    getResults() {
        return {
            file_path: this.filePath,
            is_dependency: this.is_dependency,
            functions: this.functions,
            classes: this.classes,
            variables: this.variables,
            imports: this.imports,
            exports: this.exports,
            function_calls: this.functionCalls
        };
    }
}

/**
 * Code Graph Builder for JavaScript (unchanged)
 */
class JavaScriptCodeGraph {
    constructor() {
        this.nodes = new Map();
        this.edges = new Map();
        this.files = new Map();
    }

    parseFile(filePath, sourceCode, is_dependency = false) {
        const visitor = new JavaScriptCodeVisitor(filePath, is_dependency);
        const results = visitor.parse(sourceCode);

        return results;
    }

    // Additional Function , maybe we can use in future

    addNodes(results) {
        const { filePath, functions, classes, variables, imports, exports } = results;

        functions.forEach(func => {
            const nodeId = `${filePath}:${func.name}:${func.lineNumber}`;
            this.nodes.set(nodeId, {
                id: nodeId,
                name: func.name,
                filePath: filePath,
                ...func
            });
        });

        classes.forEach(cls => {
            const nodeId = `${filePath}:${cls.name}:${cls.lineNumber}`;
            this.nodes.set(nodeId, {
                id: nodeId,
                name: cls.name,
                filePath: filePath,
                ...cls
            });
        });

        variables.forEach(variable => {
            const nodeId = `${filePath}:${variable.name}:${variable.lineNumber}`;
            this.nodes.set(nodeId, {
                id: nodeId,
                name: variable.name,
                filePath: filePath,
                ...variable
            });
        });

        [...imports, ...exports].forEach(item => {
            const nodeId = `${filePath}:${item.name}:${item.lineNumber}`;
            this.nodes.set(nodeId, {
                id: nodeId,
                name: item.name,
                filePath: filePath,
                ...item
            });
        });
    }

    addEdges(results) {
        const { filePath, functionCalls, imports, exports } = results;

        functionCalls.forEach(call => {
            const fromId = `${filePath}:${call.context || 'global'}`;
            const toId = call.name;

            this.addEdge(fromId, toId, 'calls', {
                line_number: call.lineNumber,
                args: call.args
            });
        });

        imports.forEach(imp => {
            const fromId = `${filePath}:${imp.name}:${imp.lineNumber}`;
            const toId = imp.moduleSource;

            this.addEdge(fromId, toId, 'imports', {
                type: imp.type,
                alias: imp.alias
            });
        });
    }

    addEdge(fromId, toId, type, metadata = {}) {
        const edgeId = `${fromId}->${toId}`;

        if (!this.edges.has(edgeId)) {
            this.edges.set(edgeId, []);
        }

        this.edges.get(edgeId).push({
            from: fromId,
            to: toId,
            type: type,
            ...metadata
        });
    }

    /**
     * Get all nodes
     */
    getNodes() {
        return Array.from(this.nodes.values());
    }

    /**
     * Get all edges
     */
    getEdges() {
        return Array.from(this.edges.values()).flat();
    }

    /**
     * Get code graph as JSON
     */
    toJSON() {
        return Object.fromEntries(this.files);
    }
}

// Usage example:

const fs = require('fs');
const codeGraph = new JavaScriptCodeGraph();

// Parse a JavaScript file
const inputPath = process.argv[2];

if (!inputPath) {
    console.error('No input file provided.');
    process.exit(1);
}
try {
    const sourceCode = fs.readFileSync(inputPath, 'utf8');
    const result = codeGraph.parseFile(inputPath, sourceCode);
    const resultJSON = JSON.stringify(result, null, 2);
    console.log(resultJSON);


} catch (err) {
    console.error('Error parsing file:', err.message);
    process.exit(1);
}




