"""
Math answer grading utilities vendored from SeerAttention's
eval/reasoning_tasks/Utils/{parser,grader,math_normalization}.py.

Public API:
    find_box(text)           -> str   innermost \\boxed{...} contents
    extract_answer(text)     -> str   boxed content with light cleanup
    strip_string(s)          -> str   normalize latex/units/spacing
    math_equal(pred, gold)   -> bool  numeric + symbolic equivalence
    check_is_correct(pred, gold) -> bool  strip_string both then math_equal

Multi-choice / theoremqa helpers are intentionally omitted.

Upstream license: see https://github.com/microsoft/SeerAttention
"""

from __future__ import annotations

import multiprocessing
import re
from math import isclose
from typing import Union

import regex
import sympy
from latex2sympy2 import latex2sympy
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from word2number import w2n


# ---------------------------------------------------------------------------
# strip_string + helpers (parser.py)
# ---------------------------------------------------------------------------
def _fix_fracs(string):
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
                except Exception:
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


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except Exception:
        return string


def _fix_sqrt(string):
    return re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)


def _convert_word_number(text: str) -> str:
    try:
        return str(w2n.word_to_num(text))
    except Exception:
        return text


_UNIT_TEXTS = [
    "east", "degree", "mph", "kmph", "ft", "m sqaure", " m east", "sq m", "deg",
    "mile", "q .", "monkey", "prime", "ratio", "profit of rs", "rd", "o", "gm",
    "p . m", "lb", "tile", "per", "dm", "lt", "gain", "ab", "way", "west",
    "a .", "b .", "c .", "d .", "e .", "f .", "g .", "h .", "t", "a", "h",
    "no change", "men", "soldier", "pie", "bc", "excess", "st", "inches",
    "noon", "percent", "by", "gal", "kmh", "c", "acre", "rise", "a . m", "th",
    "π r 2", "sq", "mark", "l", "toy", "coin", "sq . m", "gallon", "° f",
    "profit", "minw", "yr", "women", "feet", "am", "pm", "hr", "cu cm",
    "square", "v â € ™", "are", "rupee", "rounds", "cubic", "cc", "mtr", "s",
    "ohm", "number", "kmph", "day", "hour", "minute", "min", "second", "man",
    "woman", "sec", "cube", "mt", "sq inch", "mp", "∏ cm ³", "hectare", "more",
    "sec", "unit", "cu . m", "cm 2", "rs .", "rs", "kg", "g", "month", "km",
    "m", "cm", "mm", "apple", "liter", "loss", "yard", "pure", "year",
    "increase", "decrease", "d", "less", "Surface", "litre", "pi sq m", "s .",
    "metre", "meter", "inch",
]
_UNIT_TEXTS.extend([t + "s" for t in _UNIT_TEXTS])


def strip_string(string):
    """Vendored from SeerAttention Utils/parser.py:strip_string."""
    string = str(string).strip()
    string = string.replace("\n", "")
    string = string.rstrip(".")
    string = string.replace("\\!", "")

    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")

    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = (
        string.replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
    )

    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("\\{", "{").replace("\\}", "}")

    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        string = _string

    for unit_text in _UNIT_TEXTS:
        _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
        if _string != "":
            string = _string

    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace("$", "")
    string = string.replace("\\(", "").replace("\\)", "")

    string = _convert_word_number(string)

    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
        string = string.replace(key, "")
    string = string.replace("\\emptyset", r"{}")
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")

    string = string.replace("\\%", "").replace("\%", "").replace("%", "")

    months = r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b"
    string = re.sub(months, "", string, flags=re.IGNORECASE)

    string = string.replace(" .", " 0.").replace("{.", "{0.")

    if (
        string.startswith("{") and string.endswith("}") and string.isalnum()
        or string.startswith("(") and string.endswith(")") and string.isalnum()
        or string.startswith("[") and string.endswith("]") and string.isalnum()
    ):
        string = string[1:-1]

    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    string = string.replace("and", "").replace("\\mathbf", "")

    string = re.sub(r"\\mbox{.*?}", "", string)

    string.replace("'", "")
    string.replace('"', "")

    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0*$", r"\1", string)

    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)

    return string


# ---------------------------------------------------------------------------
# Boxed-answer extraction (parser.py)
# ---------------------------------------------------------------------------
def find_box(pred_str: str) -> str:
    """Innermost \\boxed{...} content; "" if missing."""
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a


def extract_answer(pred_str: str) -> str:
    """SeerAttention Utils/parser.py:extract_answer (data-name independent path)."""
    pred_str = pred_str.replace("ки", "")
    pred = ""
    if "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a

    pred = re.sub(r"\n\s*", "", pred)
    if pred and pred[0] == ":":
        pred = pred[1:]
    if pred and pred[-1] == ".":
        pred = pred[:-1]
    if pred and pred[-1] == "/":
        pred = pred[:-1]
    return pred


# ---------------------------------------------------------------------------
# math_equal (grader.py)
# ---------------------------------------------------------------------------
def parse_digits(num):
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except Exception:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except Exception:
                pass
    return None


def is_digit(num):
    return parse_digits(num) is not None


def str_to_pmatrix(input_str):
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []
    for m in matrix_str:
        m = m.strip("{}")
        pmatrix = r"\begin{pmatrix}" + m.replace(",", "\\") + r"\end{pmatrix}"
        pmatrix_list.append(pmatrix)
    return ", ".join(pmatrix_list)


def numeric_equal(prediction: float, reference: float):
    return isclose(reference, prediction, abs_tol=1e-4)


def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\"))
            except Exception:
                try:
                    return f(s)
                except Exception:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    try:
        if str(a) == str(b) or a == b:
            return True
    except Exception:
        pass
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except Exception:
        pass
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except Exception:
        pass
    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except Exception:
        pass
    try:
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except Exception:
        pass
    return False


def _symbolic_equal_process(a, b, output_queue):
    output_queue.put(symbolic_equal(a, b))


def call_with_timeout(func, *args, timeout=3, **kwargs):
    output_queue = multiprocessing.Queue()
    process_args = args + (output_queue,)
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        return False
    try:
        return output_queue.get_nowait()
    except Exception:
        return False


_SINGLE_CHOICE_PATTERNS = [
    r"^\(A\)", r"^\(B\)", r"^\(C\)", r"^\(D\)", r"^\(E\)",
    r"^A\.", r"^B\.", r"^C\.", r"^D\.", r"^E\.",
    r"^A\)", r"^B\)", r"^C\)", r"^D\)", r"^E\)",
    r"^\*\*A\*\*", r"^\*\*B\*\*", r"^\*\*C\*\*", r"^\*\*D\*\*", r"^\*\*E\*\*",
    r"^A:", r"^B:", r"^C:", r"^D:", r"^E:",
]


def _choice_answer_clean(pred: str):
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]
    pred = pred[-1]
    return pred.rstrip(".").rstrip("/")


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = True,
    depth: int = 0,
    max_depth: int = 5,
) -> bool:
    """Vendored from SeerAttention Utils/grader.py:math_equal."""
    if depth > max_depth:
        return False
    if prediction is None or reference is None:
        return False
    if str(prediction).strip().lower() == str(reference).strip().lower():
        return True
    if reference in ["A", "B", "C", "D", "E"] and _choice_answer_clean(str(prediction)) == reference:
        return True

    for pattern in _SINGLE_CHOICE_PATTERNS:
        if regex.match(pattern, str(prediction)):
            cleaned = regex.sub(pattern, "", str(prediction), count=1).strip()
            if math_equal(cleaned, reference, include_percentage, is_close,
                          timeout=timeout, depth=depth + 1, max_depth=max_depth):
                return True

    if "," in str(prediction) and "," in str(reference):
        pred_parts = [p.strip() for p in str(prediction).split(",")]
        ref_parts = [p.strip() for p in str(reference).split(",")]
        if len(pred_parts) == len(ref_parts):
            ps = sorted(pred_parts)
            rs = sorted(ref_parts)
            if all(
                math_equal(ps[i], rs[i], include_percentage, is_close,
                           timeout=timeout, depth=depth + 1, max_depth=max_depth)
                for i in range(len(ps))
            ):
                return True

    try:
        if is_digit(prediction) and is_digit(reference):
            p = parse_digits(prediction)
            r = parse_digits(reference)
            gt_result = [r / 100, r, r * 100] if include_percentage else [r]
            for item in gt_result:
                try:
                    if is_close:
                        if numeric_equal(p, item):
                            return True
                    else:
                        if item == p:
                            return True
                except Exception:
                    continue
            return False
    except Exception:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    reference = str(reference).strip()
    prediction = str(prediction).strip()

    if "pmatrix" in prediction and "pmatrix" not in reference:
        reference = str_to_pmatrix(reference)

    pred_str, ref_str = prediction, reference
    if (
        prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")
    ) or (
        prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str.lower() == ref_str.lower():
        return True

    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close,
                           timeout=timeout, depth=depth + 1, max_depth=max_depth)
                for i in range(len(pred_parts))
            ):
                return True

    if (
        (prediction.startswith("\\begin{pmatrix}") or prediction.startswith("\\begin{bmatrix}"))
        and (prediction.endswith("\\end{pmatrix}") or prediction.endswith("\\end{bmatrix}"))
        and (reference.startswith("\\begin{pmatrix}") or reference.startswith("\\begin{bmatrix}"))
        and (reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}"))
    ):
        pred_lines = [
            line.strip()
            for line in prediction[len("\\begin{pmatrix}") : -len("\\end{pmatrix}")].split("\\\\")
            if line.strip()
        ]
        ref_lines = [
            line.strip()
            for line in reference[len("\\begin{pmatrix}") : -len("\\end{pmatrix}")].split("\\\\")
            if line.strip()
        ]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pl, rl in zip(pred_lines, ref_lines):
                pp = pl.split("&")
                rp = rl.split("&")
                if len(pp) == len(rp):
                    if not all(
                        math_equal(pp[i], rp[i], include_percentage, is_close,
                                   timeout=timeout, depth=depth + 1, max_depth=max_depth)
                        for i in range(len(pp))
                    ):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            return True
    elif (
        prediction.count("=") == 1
        and len(prediction.split("=")[0].strip()) <= 2
        and "=" not in reference
    ):
        if math_equal(prediction.split("=")[1], reference, include_percentage, is_close,
                      timeout=timeout, depth=depth + 1, max_depth=max_depth):
            return True
    elif (
        reference.count("=") == 1
        and len(reference.split("=")[0].strip()) <= 2
        and "=" not in prediction
    ):
        if math_equal(prediction, reference.split("=")[1], include_percentage, is_close,
                      timeout=timeout, depth=depth + 1, max_depth=max_depth):
            return True

    if timeout:
        if call_with_timeout(_symbolic_equal_process, prediction, reference):
            return True
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False


def check_is_correct(pred, gt, timeout=True):
    """SeerAttention Utils/grader.py:check_is_correct."""
    return math_equal(strip_string(pred), strip_string(gt), timeout=timeout)
