import streamlit as st
import sympy as sp
import numpy as np
from sympy.parsing.sympy_parser import parse_expr
class SymbolicProcessor:
    """
    Xử lý tất cả logic tượng trưng: phân tích chuỗi, tiêu chuẩn hóa
    biến, và tạo đạo hàm.
    """
    def __init__(self, indep_var_str, dep_var_str, func_str):
        self.indep_var_str = indep_var_str
        self.dep_var_str = dep_var_str
        self.func_str = func_str
        self.t = sp.symbols('t')
        self.y = sp.symbols('y') 
        self.f_expr = None
        self.f_numeric = None
        self.deriv_exprs = None
        self.deriv_funcs = None
    def standardize_expression(self):
        """
        Chuyển đổi chuỗi hàm của người dùng (ví dụ: 'x+y') thành một
        biểu thức SymPy tiêu chuẩn hóa sử dụng (t, y).
        """
        try:
            user_t = sp.symbols(self.indep_var_str)
            user_y = sp.symbols(self.dep_var_str)
            local_dict = {
                "sin": sp.sin,
                "cos": sp.cos,
                "tan": sp.tan,
                "exp": sp.exp,
                "log": sp.log,
                "sqrt": sp.sqrt,
                "pi": sp.pi,
                "np": sp  
            }
            expr = parse_expr(self.func_str, local_dict=local_dict, global_dict=None)
            self.f_expr = expr.subs([
                (user_t, self.t),
                (user_y, self.y)
            ])
        except Exception as e:
            st.error(f"Lỗi phân tích phương trình: {e}. "
                     f"Hãy chắc chắn sử dụng cú pháp Python (ví dụ: 'y**2').")
            self.f_expr = None
    def get_numeric_function(self):
        """
        Chuyển đổi biểu thức SymPy tiêu chuẩn hóa thành một hàm Python
        nhanh, có thể gọi được bằng NumPy.
        """
        if self.f_expr is None:
            return
        try:
            self.f_numeric = sp.lambdify(
                (self.t, self.y), 
                self.f_expr, 
                modules='numpy'
            )
        except Exception as e:
            st.error(f"Lỗi khi tạo hàm số học: {e}")
            self.f_numeric = None
    def generate_total_derivatives(self, order):
        """
        Tạo ra các đạo hàm toàn phần bậc cao của f một cách tượng trưng.
        f^(k) = d/dt [f^(k-1)] = (df^(k-1)/dt) + (df^(k-1)/dy) * f
        """
        if self.f_expr is None:
            return
        self.deriv_exprs = [self.f_expr]
        try:
            for i in range(1, order):
                prev_deriv = self.deriv_exprs[-1]
                df_dt = sp.diff(prev_deriv, self.t)
                df_dy = sp.diff(prev_deriv, self.y)
                total_deriv = df_dt + df_dy * self.f_expr
                self.deriv_exprs.append(total_deriv)
            self.deriv_funcs = [
                sp.lambdify((self.t, self.y), expr, modules='numpy') 
                for expr in self.deriv_exprs
            ]
            st.success(f"Đã tạo thành công {order} đạo hàm tượng trưng.")
            with st.expander("Xem các đạo hàm tượng trưng đã được tạo"):
                st.markdown("Hệ thống đã tự động tính toán các đạo hàm toàn phần sau (sử dụng quy tắc chuỗi):")
                st.latex(f"f(t, y) = {sp.latex(self.f_expr)}")
                for i, expr in enumerate(self.deriv_exprs[1:], start=1):
                    st.latex(f"f^{{({i})}}(t, y) = \\frac{{d^{i}f}}{{dt^{i}}} = {sp.latex(expr)}")
        except Exception as e:
            st.error(f"Lỗi khi tạo đạo hàm Taylor: {e}")
            self.deriv_funcs = None