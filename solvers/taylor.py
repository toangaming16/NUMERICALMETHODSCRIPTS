import numpy as np
import pandas as pd
import streamlit as st
class TaylorSolver:
    def __init__(self, derivative_functions, order):
        self.deriv_funcs = derivative_functions
        self.order = order
    def solve(self, t0, y0, tend, N):
        t_values = np.linspace(t0, tend, N + 1)
        y_values = np.zeros(N + 1)
        y_values[0] = y0
        h = (tend - t0) / N
        columns = ["Step", "t_i", "y_i"]
        for k in range(self.order):
            columns.append(f'Term (h^{k+1})')
        columns.append('y_{i+1}')
        step_data = []
        step_explanations = []
        for i in range(N):
            ti = t_values[i]
            yi = y_values[i]
            y_next = yi
            h_power = 1.0
            factorial = 1.0
            row_data = [i, ti, yi]
            step_str = f"**Bước {i+1} (từ t = {ti:.4f} đến t = {t_values[i+1]:.4f}):**\n\n"
            step_str += f"* Giá trị ban đầu: $y_{i} = y({ti:.4f}) = {yi:.6f}$\n"
            step_str += "* Tính các số hạng (terms) của chuỗi Taylor:\n"
            terms_str_list = []
            for k in range(self.order):
                f_k_value = np.real(self.deriv_funcs[k](ti, yi))
                h_power *= h
                factorial *= (k + 1)
                term = (h_power / factorial) * f_k_value
                y_next += term
                row_data.append(term)
                step_str += f"    * Bậc {k+1}: $\\frac{{h^{k+1}}}{{(k+1)!}} f^{{({k})}}(t_i, y_i) = \\frac{{{h:.4f}^{k+1}}}{{{int(factorial)}}} ({f_k_value:.6f}) = {term:.6f}$\n"
                terms_str_list.append(f"{term:.6f}")
            y_values[i+1] = y_next
            row_data.append(y_next)
            step_data.append(row_data)
            step_str += f"\n* Cộng các số hạng để tìm $y_{i+1}$:\n"
            step_str += f"    $y_{i+1} = y_i + (Term_1) + (Term_2) + \dots$\n"
            step_str += f"    $y_{i+1} = {yi:.6f} + {' + '.join(terms_str_list)}$\n"
            step_str += f"    **$y_{i+1} = {y_next:.6f}$**\n\n---\n"
            step_explanations.append(step_str)
        with st.expander(f"Xem chi tiết tính toán từng bước của Taylor (Bậc {self.order})"):
            tab1, tab2 = st.tabs(["Giải thích từng bước", "Bảng dữ liệu chi tiết"])
            with tab1:
                if step_data:
                    tab1.markdown("\n".join(step_explanations))
                else:
                    tab1.info("Không có dữ liệu bước để hiển thị.")
            with tab2:
                if step_data:
                    df_steps = pd.DataFrame(step_data, columns=columns)
                    tab2.dataframe(df_steps.style.format("{:.6f}"))
                else:
                    tab2.info("Không có dữ liệu bước để hiển thị.")           
        return t_values, y_values