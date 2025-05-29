import os
import sys
import re
import ast
import json
import time
import subprocess
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import traceback
from pathlib import Path
from openai import OpenAI
import google.generativeai as genai
genai.configure(api_key="your_api_key")

class DefectType(Enum):
    """Comprehensive defect types for Python programs"""
    SYNTAX_ERROR = "syntax_error"
    INDENTATION_ERROR = "indentation_error"
    NAME_ERROR = "name_error"
    TYPE_ERROR = "type_error"
    INDEX_ERROR = "index_error"
    KEY_ERROR = "key_error"
    ATTRIBUTE_ERROR = "attribute_error"
    VALUE_ERROR = "value_error"
    LOGIC_ERROR = "logic_error"
    MISSING_IMPORT = "missing_import"
    MISSING_RETURN = "missing_return"
    INFINITE_LOOP = "infinite_loop"
    VARIABLE_SCOPE = "variable_scope"
    EXCEPTION_HANDLING = "exception_handling"
    ALGORITHM_ERROR = "algorithm_error"
    RECURSION_ERROR = "recursion_error"
    PERFORMANCE_ISSUE = "performance_issue"
    INPUT_OUTPUT_ERROR = "input_output_error"
    DATA_STRUCTURE_ERROR = "data_structure_error"
    BOUNDARY_CONDITION = "boundary_condition"

@dataclass
class LineDefect:
    line_number: int
    original_line: str
    defect_type: DefectType
    confidence: float
    suggested_fix: str
    context_lines: List[str]
    explanation: str
    severity: str = "medium"  

@dataclass
class RepairResult:
    original_code: str
    fixed_code: str
    line_numbers: List[int]
    defect_types: List[DefectType]
    confidence: float
    description: str = ""
    line_defects: List[LineDefect] = None
    execution_test_passed: bool = False
    syntax_valid: bool = False
    improvement_score: float = 0.0

@dataclass
class ProgramStats:
    filename: str
    total_lines: int
    code_lines: int
    comment_lines: int
    blank_lines: int
    functions: int
    classes: int
    complexity_score: float
    defects_found: int
    repair_attempts: int
    successful_repairs: int

class AlgorithmDetector:
    """Enhanced algorithm detection and analysis"""
    
    def __init__(self):
        self.algorithm_patterns = {
            # Sorting Algorithms
            'bucketsort': {
                'keywords': ['bucket', 'counts', 'enumerate', 'range'],
                'patterns': [r'counts\s*=.*\[\s*0\s*\]', r'enumerate\(.*\)'],
                'common_errors': [
                    {'pattern': r'enumerate\(arr\)', 'fix': 'enumerate(counts)', 'explanation': 'Should enumerate counts array'},
                    {'pattern': r'for.*in.*arr.*counts', 'fix': 'Separate counting and reconstruction phases'}
                ]
            },
            'mergesort': {
                'keywords': ['merge', 'divide', 'left', 'right', 'mid'],
                'patterns': [r'def\s+merge', r'left.*right', r'mid\s*='],
                'common_errors': [
                    {'pattern': r'return\s+merge\(', 'fix': 'Ensure proper recursive calls'}
                ]
            },
            'quicksort': {
                'keywords': ['pivot', 'partition', 'left', 'right'],
                'patterns': [r'pivot', r'partition', r'less.*greater'],
                'common_errors': [
                    {'pattern': r'pivot\s*=.*\[0\]', 'fix': 'Consider better pivot selection'}
                ]
            },
            'heapsort': {
                'keywords': ['heap', 'heapify', 'parent', 'child'],
                'patterns': [r'heapify', r'parent.*child', r'heap'],
                'common_errors': []
            },
            
            # Search Algorithms
            'breadth_first_search': {
                'keywords': ['bfs', 'queue', 'visited', 'neighbors'],
                'patterns': [r'queue', r'visited', r'neighbors'],
                'common_errors': [
                    {'pattern': r'queue\s*=\s*\[\]', 'fix': 'Use collections.deque for better performance'}
                ]
            },
            'depth_first_search': {
                'keywords': ['dfs', 'stack', 'visited', 'recursive'],
                'patterns': [r'stack', r'visited', r'dfs'],
                'common_errors': []
            },
            
            # Dynamic Programming
            'knapsack': {
                'keywords': ['dp', 'weight', 'value', 'capacity'],
                'patterns': [r'dp\[.*\]\[.*\]', r'weight.*value'],
                'common_errors': []
            },
            'longest_common_subsequence': {
                'keywords': ['lcs', 'dp', 'subsequence'],
                'patterns': [r'lcs', r'dp\[.*\]\[.*\]'],
                'common_errors': []
            },
            
            # Graph Algorithms
            'shortest_path': {
                'keywords': ['dijkstra', 'distance', 'vertex', 'edge'],
                'patterns': [r'distance', r'vertex', r'edge', r'graph'],
                'common_errors': []
            },
            'binary_tree': {
                'keywords': ['node', 'left', 'right', 'root'],
                'patterns': [r'class.*Node', r'left.*right', r'root'],
                'common_errors': []
            },
            'gcd': {
                'keywords': ['gcd', 'euclidean', 'remainder'],
                'patterns': [r'gcd', r'%', r'while.*!=\s*0'],
                'common_errors': []
            },
            'factorial': {
                'keywords': ['factorial', 'fact'],
                'patterns': [r'factorial', r'fact.*\*'],
                'common_errors': []
            },
            'fibonacci': {
                'keywords': ['fib', 'fibonacci'],
                'patterns': [r'fib', r'fibonacci'],
                'common_errors': []
            },
            
            # String Algorithms
            'palindrome': {
                'keywords': ['palindrome', 'reverse'],
                'patterns': [r'palindrome', r'\[::-1\]'],
                'common_errors': []
            },
            
            # Miscellaneous
            'hanoi': {
                'keywords': ['hanoi', 'tower', 'disk'],
                'patterns': [r'hanoi', r'tower', r'disk'],
                'common_errors': []
            }
        }
    
    def detect_algorithm_type(self, code: str, filename: str = "") -> Optional[str]:
        """Detect the type of algorithm based on code content and filename"""
        code_lower = code.lower()
        filename_lower = filename.lower()

        for algo_name in self.algorithm_patterns.keys():
            if algo_name in filename_lower:
                return algo_name
        best_match = None
        best_score = 0
        
        for algo_name, info in self.algorithm_patterns.items():
            score = 0
            for keyword in info['keywords']:
                if keyword in code_lower:
                    score += 2
            for pattern in info['patterns']:
                if re.search(pattern, code, re.IGNORECASE):
                    score += 3
            
            if score > best_score:
                best_score = score
                best_match = algo_name
        return best_match if best_score >= 3 else None

class PythonAnalyzer:
    """Enhanced analyzer for detecting Python code issues"""
    
    def __init__(self):
        self.error_patterns = self._initialize_error_patterns()
        self.algorithm_detector = AlgorithmDetector()
        self.builtin_functions = set(dir(__builtins__))
        self.common_imports = {
            'math', 'sys', 'os', 'collections', 'itertools', 'functools',
            'random', 'time', 'datetime', 're', 'json', 'pickle'
        }
    
    def _initialize_error_patterns(self) -> Dict[DefectType, List[Dict]]:
        """Initialize comprehensive error patterns"""
        return {
            DefectType.SYNTAX_ERROR: [
                {
                    'pattern': r'print\s+[^(]',
                    'confidence': 0.9,
                    'fix': lambda m: m.group(0).replace('print ', 'print(') + ')',
                    'explanation': 'Missing parentheses in print statement (Python 3 syntax)',
                    'severity': 'high'
                },
                {
                    'pattern': r':\s*$',
                    'confidence': 0.3,
                    'fix': 'Add code block after colon',
                    'explanation': 'Empty code block after colon',
                    'severity': 'medium'
                }
            ],
            DefectType.INDENTATION_ERROR: [
                {
                    'pattern': r'^\s{1,3}[^\s]',
                    'confidence': 0.4,
                    'fix': 'Use 4 spaces for indentation',
                    'explanation': 'Inconsistent indentation (should be 4 spaces)',
                    'severity': 'medium'
                }
            ],
            DefectType.ALGORITHM_ERROR: [
                {
                    'pattern': r'enumerate\(arr\).*counts',
                    'confidence': 0.9,
                    'fix': 'enumerate(counts) instead of enumerate(arr)',
                    'explanation': 'Wrong array being enumerated in algorithm',
                    'severity': 'high'
                }
            ],
            DefectType.INDEX_ERROR: [
                {
                    'pattern': r'(\w+)\[(\w+|\d+)\]',
                    'confidence': 0.6,
                    'fix': 'Add bounds checking',
                    'explanation': 'Potential index out of bounds error',
                    'severity': 'medium'
                }
            ],
            DefectType.MISSING_IMPORT: [
                {
                    'pattern': r'(math|sys|os|collections|random|time|re|json)\.',
                    'confidence': 0.8,
                    'fix': 'Add missing import statement',
                    'explanation': 'Using module without import',
                    'severity': 'high'
                }
            ],
            DefectType.INFINITE_LOOP: [
                {
                    'pattern': r'while\s+True:',
                    'confidence': 0.5,
                    'fix': 'Add break condition',
                    'explanation': 'Potential infinite loop',
                    'severity': 'medium'
                }
            ],
            DefectType.RECURSION_ERROR: [
                {
                    'pattern': r'def\s+(\w+).*\1\(',
                    'confidence': 0.6,
                    'fix': 'Add base case',
                    'explanation': 'Recursive function may lack base case',
                    'severity': 'medium'
                }
            ]
        }
    
    def analyze_code_structure(self, code: str, filename: str = "") -> Tuple[List[LineDefect], ProgramStats]:
        """Enhanced analysis including comprehensive checks"""
        defects = []
        lines = code.split('\n')
        stats = self._calculate_program_stats(code, filename)
        syntax_defects = self._analyze_syntax(code, lines)
        defects.extend(syntax_defects)
        algorithm_type = self.algorithm_detector.detect_algorithm_type(code, filename)
        if algorithm_type:
            algo_defects = self._analyze_algorithm_specific(code, lines, algorithm_type)
            defects.extend(algo_defects)
        pattern_defects = self._analyze_patterns(lines)
        defects.extend(pattern_defects)
        semantic_defects = self._analyze_semantics(code, lines)
        defects.extend(semantic_defects)
        performance_defects = self._analyze_performance(lines)
        defects.extend(performance_defects)
        
        stats.defects_found = len(defects)
        return defects, stats
    
    def _calculate_program_stats(self, code: str, filename: str) -> ProgramStats:
        """Calculate comprehensive program statistics"""
        lines = code.split('\n')
        
        total_lines = len(lines)
        code_lines = 0
        comment_lines = 0
        blank_lines = 0
        functions = 0
        classes = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank_lines += 1
            elif stripped.startswith('#'):
                comment_lines += 1
            else:
                code_lines += 1
                if stripped.startswith('def '):
                    functions += 1
                elif stripped.startswith('class '):
                    classes += 1
        
        # Calculate complexity score (simple heuristic)
        complexity_score = (
            functions * 2 + 
            classes * 3 + 
            code.count('if ') + 
            code.count('for ') + 
            code.count('while ') +
            code.count('try:') +
            code.count('except')
        ) / max(code_lines, 1)
        
        return ProgramStats(
            filename=filename,
            total_lines=total_lines,
            code_lines=code_lines,
            comment_lines=comment_lines,
            blank_lines=blank_lines,
            functions=functions,
            classes=classes,
            complexity_score=complexity_score,
            defects_found=0,  # Will be set later
            repair_attempts=0,
            successful_repairs=0
        )
    
    def _analyze_syntax(self, code: str, lines: List[str]) -> List[LineDefect]:
        """Analyze syntax errors"""
        defects = []
        
        try:
            ast.parse(code)
        except SyntaxError as e:
            line_number = e.lineno - 1 if e.lineno else 0
            original_line = lines[line_number] if line_number < len(lines) else ""
            defects.append(LineDefect(
                line_number=line_number,
                original_line=original_line,
                defect_type=DefectType.SYNTAX_ERROR,
                confidence=1.0,
                suggested_fix=f"Fix syntax error: {e.msg}",
                context_lines=self._get_context(lines, line_number),
                explanation=f"Syntax error: {e.msg}",
                severity="critical"
            ))
        except IndentationError as e:
            line_number = e.lineno - 1 if e.lineno else 0
            original_line = lines[line_number] if line_number < len(lines) else ""
            defects.append(LineDefect(
                line_number=line_number,
                original_line=original_line,
                defect_type=DefectType.INDENTATION_ERROR,
                confidence=1.0,
                suggested_fix=f"Fix indentation: {e.msg}",
                context_lines=self._get_context(lines, line_number),
                explanation=f"Indentation error: {e.msg}",
                severity="high"
            ))
        
        return defects
    
    def _analyze_algorithm_specific(self, code: str, lines: List[str], algorithm_type: str) -> List[LineDefect]:
        """Analyze algorithm-specific issues"""
        defects = []
        algo_info = self.algorithm_detector.algorithm_patterns.get(algorithm_type, {})
        
        for error_pattern in algo_info.get('common_errors', []):
            pattern = error_pattern['pattern']
            for i, line in enumerate(lines):
                if re.search(pattern, line):
                    defects.append(LineDefect(
                        line_number=i,
                        original_line=line,
                        defect_type=DefectType.ALGORITHM_ERROR,
                        confidence=0.9,
                        suggested_fix=error_pattern['fix'],
                        context_lines=self._get_context(lines, i),
                        explanation=f"{algorithm_type}: {error_pattern['explanation']}",
                        severity="high"
                    ))
        
        return defects
    
    def _analyze_patterns(self, lines: List[str]) -> List[LineDefect]:
        """Analyze pattern-based defects"""
        defects = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            
            # Check for common patterns
            for defect_type, patterns in self.error_patterns.items():
                for pattern_info in patterns:
                    if re.search(pattern_info['pattern'], line):
                        suggested_fix = pattern_info['fix']
                        if callable(suggested_fix):
                            match = re.search(pattern_info['pattern'], line)
                            suggested_fix = suggested_fix(match) if match else "Fix pattern"
                        
                        defects.append(LineDefect(
                            line_number=i,
                            original_line=line,
                            defect_type=defect_type,
                            confidence=pattern_info['confidence'],
                            suggested_fix=suggested_fix,
                            context_lines=self._get_context(lines, i),
                            explanation=pattern_info['explanation'],
                            severity=pattern_info.get('severity', 'medium')
                        ))
        
        return defects
    
    def _analyze_semantics(self, code: str, lines: List[str]) -> List[LineDefect]:
        """Analyze semantic issues"""
        defects = []
        
        # Find undefined variables
        try:
            tree = ast.parse(code)
            analyzer = VariableAnalyzer()
            analyzer.visit(tree)
            
            for var, line_num in analyzer.undefined_vars:
                if line_num < len(lines):
                    defects.append(LineDefect(
                        line_number=line_num,
                        original_line=lines[line_num],
                        defect_type=DefectType.NAME_ERROR,
                        confidence=0.8,
                        suggested_fix=f"Define variable '{var}' before use",
                        context_lines=self._get_context(lines, line_num),
                        explanation=f"Variable '{var}' used before definition",
                        severity="high"
                    ))
        except:
            pass  # Skip if AST parsing fails
        
        return defects
    
    def _analyze_performance(self, lines: List[str]) -> List[LineDefect]:
        """Analyze performance issues"""
        defects = []
        
        for i, line in enumerate(lines):
            # Check for inefficient string concatenation
            if '+=' in line and 'str' in line.lower():
                defects.append(LineDefect(
                    line_number=i,
                    original_line=line,
                    defect_type=DefectType.PERFORMANCE_ISSUE,
                    confidence=0.6,
                    suggested_fix="Use list and join() for string concatenation",
                    context_lines=self._get_context(lines, i),
                    explanation="Inefficient string concatenation in loop",
                    severity="low"
                ))
            
            # Check for list operations that could be sets
            if 'in' in line and '[' in line and ']' in line:
                defects.append(LineDefect(
                    line_number=i,
                    original_line=line,
                    defect_type=DefectType.PERFORMANCE_ISSUE,
                    confidence=0.4,
                    suggested_fix="Consider using set for membership testing",
                    context_lines=self._get_context(lines, i),
                    explanation="List membership testing can be slow",
                    severity="low"
                ))
        
        return defects
    
    def _get_context(self, lines: List[str], line_num: int, context_size: int = 3) -> List[str]:
        """Get context lines around a specific line"""
        start = max(0, line_num - context_size)
        end = min(len(lines), line_num + context_size + 1)
        return lines[start:end]

class VariableAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze variable usage"""
    
    def __init__(self):
        self.defined_vars = set()
        self.undefined_vars = []
        self.current_line = 0
    
    def visit(self, node):
        if hasattr(node, 'lineno'):
            self.current_line = node.lineno - 1
        super().visit(node)
    
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.defined_vars.add(node.id)
        elif isinstance(node.ctx, ast.Load):
            if node.id not in self.defined_vars and node.id not in dir(__builtins__):
                self.undefined_vars.append((node.id, self.current_line))
        self.generic_visit(node)

class CodeExecutor:
    """Execute code safely to test repairs"""
    
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
    
    def test_execution(self, code: str) -> Tuple[bool, str]:
        """Safely test code execution"""
        try:
            # Create a temporary file
            temp_file = "temp_test_code.py"
            with open(temp_file, 'w') as f:
                f.write(code)
            
            # Execute with timeout
            result = subprocess.run([
                sys.executable, temp_file
            ], capture_output=True, text=True, timeout=self.timeout)
            
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            success = result.returncode == 0
            output = result.stdout if success else result.stderr
            
            return success, output
            
        except subprocess.TimeoutExpired:
            return False, "Execution timeout"
        except Exception as e:
            return False, f"Execution error: {str(e)}"

class LLMRepairAgent:
    """Enhanced LLM repair agent for code fixing"""
    
    def __init__(self, model_name: str = "microsoft/phi-4-mini-instruct"):
        self.model_name = model_name
        self.analyzer = PythonAnalyzer()
        self.executor = CodeExecutor()
    
    def generate_repair_prompt(self, code: str, defects: List[LineDefect], 
                              stats: ProgramStats, filename: str = "") -> str:
        """Generate comprehensive repair prompt with enhanced context"""
        
        # Analyze defects by type and severity
        defect_summary = self._analyze_defects(defects)
        
        # Get algorithm-specific guidance
        algorithm_guidance = self._get_algorithm_guidance(code, filename)
        
        # Get complexity assessment
        complexity_info = self._get_complexity_info(stats)
        
        prompt = f"""You are an expert Python developer. Fix this broken code to make it correct and functional.

PROGRAM ANALYSIS:
- File: {filename}
- Lines of code: {stats.code_lines}
- Functions: {stats.functions}
- Classes: {stats.classes}
- Complexity score: {stats.complexity_score:.2f}

BROKEN CODE:
```python
{code}
```

{defect_summary}

{algorithm_guidance}

{complexity_info}

REPAIR REQUIREMENTS:
1. Fix all critical and high-severity issues first
2. Ensure syntactic correctness
3. Maintain algorithmic correctness
4. Handle edge cases appropriately
5. Follow Python best practices
6. Add necessary imports
7. Ensure proper error handling where needed

COMMON PATTERNS TO CHECK:
- Python 3 syntax (print statements, integer division)
- Proper indentation (4 spaces)
- Variable initialization before use
- Array/list bounds checking
- Import statements for used modules
- Algorithm-specific logic errors

Return ONLY the corrected Python code without explanation:

```python"""
        
        return prompt
    
    def _analyze_defects(self, defects: List[LineDefect]) -> str:
        """Analyze and categorize defects"""
        if not defects:
            return "No specific defects detected, but code may still need improvements.\n"
        
        defects_by_severity = defaultdict(list)
        defects_by_type = defaultdict(list)
        
        for defect in defects:
            defects_by_severity[defect.severity].append(defect)
            defects_by_type[defect.defect_type].append(defect)
        
        summary = "DETECTED ISSUES:\n"
        
        # Group by severity
        for severity in ['critical', 'high', 'medium', 'low']:
            if severity in defects_by_severity:
                summary += f"\n{severity.upper()} PRIORITY:\n"
                for defect in defects_by_severity[severity]:
                    summary += f"- Line {defect.line_number + 1}: {defect.explanation}\n"
                    summary += f"  Current: {defect.original_line.strip()}\n"
                    summary += f"  Fix: {defect.suggested_fix}\n"
        
        return summary + "\n"
    
    def _get_algorithm_guidance(self, code: str, filename: str) -> str:
        """Get enhanced algorithm-specific guidance"""
        detector = AlgorithmDetector()
        algorithm_type = detector.detect_algorithm_type(code, filename)
        
        if not algorithm_type:
            return ""
        
        guidance_map = {
            'bucketsort': """
BUCKET SORT ALGORITHM GUIDANCE:
This is a bucket sort implementation. Key components:

1. INITIALIZATION: counts = [0] * k (where k is range of values)
2. COUNTING PHASE: for x in arr: counts[x] += 1
3. RECONSTRUCTION: for i, count in enumerate(counts):
   - 'i' is the value, 'count' is frequency
   - Add value 'i' to result 'count' times
   - Use: result.extend([i] * count)

CRITICAL BUG: enumerate(arr) should be enumerate(counts)
""",
            'mergesort': """
MERGE SORT ALGORITHM GUIDANCE:
Divide-and-conquer sorting algorithm:

1. BASE CASE: if len(arr) <= 1: return arr
2. DIVIDE: mid = len(arr) // 2
3. RECURSION: left = mergesort(arr[:mid]), right = mergesort(arr[mid:])
4. MERGE: Combine sorted left and right arrays
""",
            'breadth_first_search': """
BFS ALGORITHM GUIDANCE:
Use queue for level-by-level traversal:

1. Use collections.deque for efficient queue operations
2. Mark nodes as visited before adding to queue
3. Process all neighbors of current node
""",
            'depth_first_search': """
DFS ALGORITHM GUIDANCE:
Use stack or recursion for deep traversal:

1. Mark node as visited at start
2. Recursively visit unvisited neighbors
3. Add base case for recursion termination
"""
        }
        
        return guidance_map.get(algorithm_type, f"\n{algorithm_type.upper()} ALGORITHM DETECTED\n")
    
    def _get_complexity_info(self, stats: ProgramStats) -> str:
        """Get complexity assessment and suggestions"""
        if stats.complexity_score < 2:
            return "COMPLEXITY: Low - Simple program structure"
        elif stats.complexity_score < 5:
            return "COMPLEXITY: Medium - Moderate control flow"
        else:
            return "COMPLEXITY: High - Complex logic, ensure all paths are tested"
    
    def call_llm(self, prompt: str, max_retries: int = 3) -> str:
        # for attempt in range(max_retries):
        #     try:
        #         time.sleep(1.5) 
        #         client = OpenAI(
        #             base_url="https://integrate.api.nvidia.com/v1",
        #             api_key="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"
        #         )

        #         completion = client.chat.completions.create(
        #             model=self.model_name,
        #             messages=[{"role": "user", "content": prompt}],
        #             temperature=0.1,
        #             top_p=0.7,
        #             max_tokens=2000, 
        #             stream=True
        #         )

        #         output = ""
        #         for chunk in completion:
        #             if chunk.choices[0].delta.content is not None:
        #                 output += chunk.choices[0].delta.content

        #         if output:
        #             return output.strip()

        #     except Exception as e:
        #         print(f"LLM API error (attempt {attempt + 1}): {e}")
        #         if attempt < max_retries - 1:
        #             wait_time = 2 ** attempt
        #             print(f"Waiting {wait_time} seconds before retry...")
        #             time.sleep(wait_time)

        # return ""
        for attempt in range(max_retries):
            try:
                time.sleep(10)  
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                generation_config = {
                    'temperature': 0.05, 
                    'top_p': 0.8,
                    'top_k': 20,
                    'max_output_tokens': 4000,
                }
                
                response = model.generate_content(prompt, generation_config=generation_config)
                
                if response and response.text:
                    return response.text.strip()
                    
            except Exception as e:
                print(f"LLM API error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
        
        print("LLM failed after all retry attempts")
        return ""


    def extract_code_from_response(self, response: str) -> str:
        response = re.sub(r'```python\s*', '', response)
        response = re.sub(r'```\s*$', '', response)
        
        lines = response.split('\n')
        start_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (stripped.startswith(('import ', 'from ', 'def ', 'class ', 'if ', 'for ', 'while ')) or
                re.match(r'^[a-zA-Z_]\w*\s*=', stripped) or
                stripped.startswith(('try:', 'with ', '#')) or
                stripped and not stripped.startswith(('The ', 'Here ', 'This '))):
                start_idx = i
                break
        
        # Find end of code (remove trailing explanations)
        end_idx = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if line and not line.startswith(('```', 'Note:', 'The ', 'This ')):
                end_idx = i + 1
                break
        
        code = '\n'.join(lines[start_idx:end_idx]).strip()
        return code
    
    def repair_code(self, code: str, filename: str = "") -> List[RepairResult]:
        """Repair code with comprehensive analysis"""
        results = []
        max_attempts = 3
        
        print(f"🔍 Analyzing code structure for {filename}...")
        
        defects, stats = self.analyzer.analyze_code_structure(code, filename)
        
        print(f"🎯 Found {len(defects)} potential issues:")
        for defect in defects:
            print(f"  Line {defect.line_number + 1}: {defect.defect_type.value} (confidence: {defect.confidence:.2f})")
            print(f"    {defect.explanation}")
        
        # Generate repairs
        for attempt in range(max_attempts):
            print(f"🔧 Repair attempt {attempt + 1}/{max_attempts}...")
            
            # Generate repair prompt
            repair_prompt = self.generate_repair_prompt(code, defects, stats, filename)
            if attempt > 0:
                retry_guidance = f"""
PREVIOUS ATTEMPT FAILED - TRY DIFFERENT APPROACH:
- Double-check the algorithm logic
- Consider alternative solutions
- Verify variable names match intended use

Attempt #{attempt + 1}:
"""
                repair_prompt = retry_guidance + repair_prompt
            
            fixed_code_response = self.call_llm(repair_prompt)
            if not fixed_code_response:
                print(f"No response from LLM on attempt {attempt + 1}")
                continue
            fixed_code = self.extract_code_from_response(fixed_code_response)
            if not fixed_code:
                print(f"No valid code extracted on attempt {attempt + 1}")
                continue
                
            try:
                ast.parse(fixed_code)
                syntax_valid = True
            except SyntaxError as e:
                print(f"Generated code has syntax errors on attempt {attempt + 1}: {e}")
                syntax_valid = False
                continue
            
            if fixed_code.strip() == code.strip():
                print(f"⚠️ Generated code identical to original on attempt {attempt + 1}")
                continue
            
            # Test execution
            execution_success, execution_output = self.executor.test_execution(fixed_code)
            confidence = sum(d.confidence for d in defects) / len(defects) if defects else 0.5
            
            repair_result = RepairResult(
                original_code=code,
                fixed_code=fixed_code,
                line_numbers=[d.line_number for d in defects],
                defect_types=[d.defect_type for d in defects],
                confidence=confidence,
                description=f"Repair attempt {attempt + 1} addressing {len(defects)} issues",
                line_defects=defects,
                execution_test_passed=execution_success,
                syntax_valid=syntax_valid,
                improvement_score=self._calculate_improvement_score(defects, syntax_valid, execution_success)
            )
            
            results.append(repair_result)
            print(f"Generated repair attempt {attempt + 1} - Syntax: {'✓' if syntax_valid else '✗'}, Execution: {'✓' if execution_success else '✗'}")
        
        return results
    
    def _calculate_improvement_score(self, defects: List[LineDefect], syntax_valid: bool, execution_success: bool) -> float:
        score = 0.0
        if syntax_valid:
            score += 0.4
        if execution_success:
            score += 0.4
        if defects:
            high_severity_count = sum(1 for d in defects if d.severity in ['critical', 'high'])
            defect_score = min(0.2, high_severity_count * 0.05)
            score += defect_score
        
        return min(1.0, score)

class PythonCodeCorrector:
    """Main class for Python code correction"""
    
    def __init__(self, output_dir: str = "repaired_code"):
        self.repair_agent = LLMRepairAgent()
        self.output_dir = output_dir
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"📁 Created output directory: {self.output_dir}")
    
    def process_program(self, program_file: str) -> List[RepairResult]:
        """Process a single Python program"""
        try:
            with open(program_file, 'r') as f:
                code = f.read()
            
            filename = os.path.basename(program_file)
            print(f"\n{'='*70}")
            print(f"🔧 Processing {filename}")
            print(f"{'='*70}")
            try:
                ast.parse(code)
                print("Original code has valid syntax")
            except SyntaxError as e:
                print(f" Original code has syntax errors: {e}")
            
            repairs = self.repair_agent.repair_code(code, filename)
            if not repairs:
                print("No repairs generated")
                return []
            base_name = os.path.splitext(filename)[0]
            for i, repair in enumerate(repairs):
                if i == 0:
                    original_file = os.path.join(self.output_dir, f"{base_name}_original.py")
                    print(f"💾 Saved original to {original_file}")
                repair_file = os.path.join(self.output_dir, f"{base_name}_repair_{i+1}.py")
                with open(repair_file, 'w') as f:
                    f.write(repair.fixed_code)
                print(f"💾 Saved repair {i+1} to {repair_file}")
                metadata_file = os.path.join(self.output_dir, f"{base_name}_repair_{i+1}_metadata.json")
                metadata = {
                    'original_file': program_file,
                    'repair_attempt': i + 1,
                    'defects_found': len(repair.line_defects) if repair.line_defects else 0,
                    'defect_types': [dt.value for dt in repair.defect_types],
                    'confidence': repair.confidence,
                    'description': repair.description,
                    'line_numbers_affected': repair.line_numbers,
                    'syntax_valid': repair.syntax_valid,
                    'execution_test_passed': repair.execution_test_passed,
                    'improvement_score': repair.improvement_score
                }
            
            return repairs
            
        except Exception as e:
            print(f"Error processing {program_file}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def process_directory(self, programs_dir: str) -> Dict:
        """Process all Python files in a directory"""
        results = {
            'total_programs': 0,
            'processed_programs': 0,
            'failed_programs': 0,
            'total_repairs': 0,
            'successful_repairs': 0,
            'details': []
        }
        
        if not os.path.exists(programs_dir):
            print(f"Programs directory not found: {programs_dir}")
            return results
        
        program_files = [f for f in os.listdir(programs_dir) if f.endswith('.py')]
        print(f"📁 Found {len(program_files)} Python files in {programs_dir}")
        
        for program_file in program_files:
            program_path = os.path.join(programs_dir, program_file)
            repairs = self.process_program(program_path)
            
            results['total_programs'] += 1
            
            if repairs:
                results['processed_programs'] += 1
                results['total_repairs'] += len(repairs)
                successful_count = sum(1 for r in repairs if r.syntax_valid and r.improvement_score > 0.5)
                results['successful_repairs'] += successful_count
                
                status = 'processed'
            else:
                results['failed_programs'] += 1
                status = 'failed'
            
            results['details'].append({
                'program': program_file,
                'status': status,
                'repairs_generated': len(repairs),
                'successful_repairs': sum(1 for r in repairs if r.syntax_valid and r.improvement_score > 0.5) if repairs else 0
            })
        
        return results
    
    def generate_report(self, results: Dict) -> str:
        """Generate processing report"""
        total = results['total_programs']
        
        report = f"""
{'='*80}
🤖 PYTHON CODE CORRECTION REPORT
{'='*80}

📊 SUMMARY:
   Total Programs: {total}
   ✅ Successfully Processed: {results['processed_programs']}
   ❌ Failed to Process: {results['failed_programs']}
   🔧 Total Repairs Generated: {results['total_repairs']}
   ⭐ High-Quality Repairs: {results['successful_repairs']}
   📁 Output Directory: {self.output_dir}

📋 DETAILED RESULTS:
"""    
        for detail in results['details']:
            if detail['status'] == 'processed':
                icon = f"PROCESSED ({detail['repairs_generated']} repairs, {detail['successful_repairs']} high-quality)"
            else:
                icon = "FAILED TO PROCESS"
            
            report += f"   {icon} - {detail['program']}\n"
        
        report += f"\n{'='*80}\n"
        return report
if __name__ == "__main__":
    print("🚀 Starting Python Code Corrector")
    print("=" * 60)
    programs_dir = "/Users/akhileshkumar/Desktop/aims-dtu/python_programs"
    output_dir = "repaired_codee"
    
    corrector = PythonCodeCorrector(output_dir=output_dir)
    
    print(f"📁 Programs directory: {programs_dir}")
    print(f"💾 Output directory: {output_dir}")
    results = corrector.process_directory(programs_dir)

    if results['total_programs'] > 0:
        print(f"\nFINAL STATISTICS:")
        print(f"   Programs Processed: {results['total_programs']}")
        print(f"   Successful: {results['processed_programs']}")
        print(f"   Failed: {results['failed_programs']}")
        print(f"   Total Repairs Generated: {results['total_repairs']}")
        print(f"   High-Quality Repairs: {results['successful_repairs']}")
        print(f"   Average Repairs per Program: {results['total_repairs']/results['total_programs']:.1f}")
        if results['total_repairs'] > 0:
            print(f"   Success Rate: {results['successful_repairs']/results['total_repairs']*100:.1f}%")
