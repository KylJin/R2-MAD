import re
import regex
from math import isclose
from abc import ABC, abstractmethod

try:
    from sympy import simplify, parse_expr
    from latex2sympy2 import latex2sympy
    LATEX2SYMPY_AVAILABLE = True
except ImportError:
    LATEX2SYMPY_AVAILABLE = False


class BaseChecker(ABC):
    @abstractmethod
    def check_answer(self, pred_answer: str, expected_answer: str) -> float:
        pass

    @abstractmethod
    def parse_answer(self, response: str) -> str:
        pass


class MATHChecker(BaseChecker):
    def check_answer(self, pred_answer: str, expected_answer: str) -> bool:
        try:
            is_correct = self._math_equal(pred_answer, expected_answer)
            return 1.0 if is_correct else 0.0
        except Exception as e:
            print(f"Error in evaluation: {e}")
            print(f"Prediction: {pred_answer}")
            print(f"Ground Truth: {expected_answer}")
    
    def parse_answer(self, response: str) -> str:
        def _find_box(string:str) -> str:
            res = regex.findall(r"\\boxed\{(.*)\}", string)
            if not res:
                res = regex.findall(r"\\fbox\{(.*)\}", string)
            if not res:
                return None
            return res[-1]
        
        if response is None:
            return ""
        
        box_format_ans = _find_box(response)
        if box_format_ans:
            return box_format_ans
        
        gsm8k_format_ans = re.findall(r"####\s*(-?[0-9\.\,]+)", response)
        if gsm8k_format_ans:
            return gsm8k_format_ans[-1].replace(",", "").replace("$", "")
        
        # Then try to find "The answer is X" format
        answer_pattern = re.findall(r'(?:[Tt]he(?:\s+final)?(?:\s+answer)?(?:\s+is)?:?)\s*([^\n]+)', response)
        if answer_pattern:
            return answer_pattern[-1]

        ans_line = response.split("\n")[-1]
        return ans_line
    
    def _math_equal(self, pred_answer: str, expected_answer: str) -> bool:
        if pred_answer is None or expected_answer is None:
            return False
        
        pred_raw = str(pred_answer)
        reference_raw = str(expected_answer)
        
        pm_pattern = r"(.+?)\s*\\pm\s*(.+)"
        
        def _check_pm_match(pm_str, list_str):
            pm_match = re.match(pm_pattern, pm_str.strip())
            if not pm_match:
                return False
            
            parts = re.split(r'[,;]', list_str)
            if len(parts) != 2:
                return False
            
            A = pm_match.group(1).strip()
            B = pm_match.group(2).strip()
            
            pm_set = {strip_string(f"{A}+{B}"), strip_string(f"{A}-{B}")}
            list_set = {strip_string(parts[0]), strip_string(parts[1])}
            
            return pm_set == list_set
    
        if r"\pm" in reference_raw and _check_pm_match(reference_raw, pred_raw):
            return True
        if r"\pm" in pred_raw and _check_pm_match(pred_raw, reference_raw):
            return True

        # Apply enhanced normalization to both prediction and reference
        prediction_str = strip_string(pred_raw)
        reference_str = strip_string(reference_raw)
    
        # Path 0: Direct string equality after normalization
        if prediction_str == reference_str:
            return True

        # Path 1: Numerical comparison
        if is_digit(prediction_str) and is_digit(reference_str):
            pred_float = parse_digits(prediction_str)
            ref_float = parse_digits(reference_str)
            if pred_float is not None and ref_float is not None and isclose(pred_float, ref_float, rel_tol=1e-4):
                return True

        # Path 2: Unordered Tuple/Vector comparison
        pred_parts = [p.strip() for p in re.split(r'[,;]', prediction_str.strip('()[]{}')) if p.strip()]
        ref_parts = [r.strip() for r in re.split(r'[,;]', reference_str.strip('()[]{}')) if r.strip()]
        if len(pred_parts) > 1 and len(pred_parts) == len(ref_parts):
            # Sort the parts before comparing to handle unordered lists
            if sorted(pred_parts) == sorted(ref_parts):
                return True

        # Path 3: Symbolic comparison
        if symbolic_equal(prediction_str, reference_str):
            return True

        return False


class MMLUProChecker(BaseChecker):
    def check_answer(self, pred_answer: str, expected_answer: str) -> bool:
        pred_char = self._normalize_char(pred_answer)
        correct_char = self._normalize_char(expected_answer)

        return 1.0 if pred_char == correct_char else 0.0
    
    def parse_answer(self, response: str) -> str:
        matches = re.findall(r"\(\(([Ａ-ＪA-Ja-j])\)\)", response)

        if len(matches) == 0:
            matches = re.findall(r"\(([Ａ-ＪA-Ja-j])\)", response)
        
        if len(matches) == 0:
            return ""
        return matches[-1].lower()
    
    def _normalize_char(self, char: str) -> str:
        if len(char) == 1 and 0xFF01 <= ord(char) <= 0xFF5E:
            char = chr(ord(char) - 0xFEE0)
        
        char = char.replace("：", ":")
        
        return char.strip().lower()[0] if char else ""


# ============================================
# Utility functions
# ============================================


ANSWER_CHECKER = {
    "MATH500": MATHChecker(),
    "MMLUPro_Engineering": MMLUProChecker(),
    "MMLUPro_Economics": MMLUProChecker(),
    "TruthfulQA": MMLUProChecker(),
}


def strip_string(string: str) -> str:
    """Enhanced string normalization to handle units, LaTeX commands, and other inconsistencies."""
    string = str(string).strip()
    
    # Remove LaTeX wrappers for text/units
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    string = re.sub(r"\\mbox\{(.*?)\}", r"\1", string)

    # Normalize and remove common symbols and units
    string = string.replace("^{\\circ}", "")
    string = string.replace("\\circ", "")
    string = string.replace("°", "")
    string = string.replace("\\%", "")
    string = string.replace("%", "")
    string = re.sub(r"\s*degrees", "", string, flags=re.IGNORECASE)
    
    # Remove common textual units (can be expanded)
    string = re.sub(r"\s*inches\^2", "", string, flags=re.IGNORECASE)
    string = re.sub(r"\s*square inches", "", string, flags=re.IGNORECASE)

    # Standard replacements
    string = string.replace("\n", "")
    if string.endswith("."):
        string = string[:-1]
    string = string.replace("\\!", "")
    string = string.replace("\\ ", " ")
    string = string.replace("\\\\", "\\")
    string = string.replace("\\\n", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    
    # Remove dollar signs
    string = string.replace("$", "")
    
    # Remove spaces and special characters last for clean comparison
    string = re.sub(r" ", "", string)
    string = re.sub("\u200b", "", string)  # Zero-width space

    string = _fix_fracs(string)
    string = _fix_sqrt(string)
    
    return string.strip()


def _fix_fracs(string):
    """Fix fraction formatting in LaTeX strings."""
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_sqrt(string):
    """Fix square root formatting in LaTeX strings."""
    return re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)


def parse_digits(num):
    """Parse numeric values from strings."""
    num_str = strip_string(str(num))
    num_str = regex.sub(",", "", num_str)
    try:
        return float(num_str)
    except:
        return None


def is_digit(num):
    """Check if a value can be parsed as a number."""
    return parse_digits(num) is not None


def symbolic_equal(a, b):
    """Check if two expressions are symbolically equal."""
    if not LATEX2SYMPY_AVAILABLE:
        try:
            return simplify(a) == simplify(b)
        except:
            return False

    def _parse(s):
        try:
            return latex2sympy(s)
        except:
            try:
                return parse_expr(s)
            except:
                return s
    
    try:
        if simplify(_parse(a) - _parse(b)) == 0:
            return True
    except:
        pass
    return False
