import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
from utils.symbolic import SymbolicProcessor
from utils.plotting import create_plot, create_results_dataframe
from solvers.taylor import TaylorSolver
from solvers.rk import RKF45Solver
from solvers.multistep import ABM4Solver
st.set_page_config(layout="wide", page_title="Máy tính ODE Nâng cao")
st.title("Máy tính Phương pháp Số cho ODE Bậc 1")
st.markdown("Một công cụ để giải và minh họa các phương pháp số cho các bài toán giá trị ban đầu (IVP) $y' = f(t,y)$.")
main_col, settings_col = st.columns([2, 1])
with settings_col:
    st.subheader("Cài đặt Bài toán")
    with st.form("ode_form"):
        st.markdown("**1. Định nghĩa Phương trình**")
        col1, col2 = st.columns(2)
        with col1:
            indep_var = st.text_input("Biến độc lập", "t")
        with col2:
            dep_var = st.text_input("Biến phụ thuộc", "y")
        func_str = st.text_input(f"Nhập hàm f({indep_var}, {dep_var}):", f"{dep_var} - {indep_var}**2 + 1")
        st.caption(f"Ví dụ: `-0.02*({dep_var} - 20)`")
        st.markdown("**2. Điều kiện Ban đầu và Khoảng**")
        col1, col2 = st.columns(2)
        with col1:
            t0 = st.number_input(f"Giá trị {indep_var} ban đầu (t0)", value=0.0)
        with col2:
            y0 = st.number_input(f"Giá trị {dep_var} ban đầu (y0)", value=0.5)
        col1, col2 = st.columns(2)
        with col1:
            tend = st.number_input(f"Giá trị {indep_var} kết thúc (tend)", value=2.0)
        with col2:
            N = st.number_input("Số bước (N)", value=10, min_value=1, step=1)
        exact_sol_str = st.text_input("Giải pháp giải tích (tùy chọn)", f"({indep_var}+1)**2 - 0.5*np.exp({indep_var})")
        st.caption(f"Dùng 'np.' cho hàm, ví dụ: `np.exp({indep_var})`")
        st.markdown("**3. Lựa chọn Phương pháp**")
        method_options = ["Phương pháp Taylor (Bậc n)", "Runge-Kutta-Fehlberg (RKF45)", "Adams-Bashforth-Moulton (ABM4)"]
        method = st.selectbox("Chọn phương pháp giải:", options=method_options)
        taylor_order = 4
        if method == "Phương pháp Taylor (Bậc n)":
            taylor_order = st.slider("Chọn bậc Taylor (n)", 1, 10, 4)
        submitted = st.form_submit_button("Giải Phương trình")
if submitted:
    with main_col:
        st.subheader("Kết quả Phân tích")
        processor = SymbolicProcessor(indep_var, dep_var, func_str)
        processor.standardize_expression()
        processor.get_numeric_function()
        if processor.f_numeric is None:
            st.error("Không thể tiếp tục. Vui lòng sửa lỗi phương trình.")
        else:
            solver = None
            t_values = np.array([])
            y_values = np.array([])
            solutions_data = {}
            try:
                if method == "Phương pháp Taylor (Bậc n)":
                    st.markdown(f"### Đang chạy: Phương pháp Taylor Bậc {taylor_order}")
                    processor.generate_total_derivatives(taylor_order)
                    if processor.deriv_funcs:
                        solver = TaylorSolver(processor.deriv_funcs, taylor_order)
                        t_values, y_values = solver.solve(t0, y0, tend, N)
                        solutions_data[f'y_Taylor(Bậc {taylor_order})'] = y_values
                elif method == "Runge-Kutta-Fehlberg (RKF45)":
                    st.markdown("### Đang chạy: Runge-Kutta-Fehlberg (RKF45)")
                    solver = RKF45Solver(processor.f_numeric)
                    t_values, y_values = solver.solve(t0, y0, tend, N)
                    solutions_data['y_RKF45'] = y_values
                elif method == "Adams-Bashforth-Moulton (ABM4)":
                    st.markdown("### Đang chạy: Adams-Bashforth-Moulton (ABM4)")
                    solver = ABM4Solver(processor.f_numeric)
                    t_values, y_values = solver.solve(t0, y0, tend, N)
                    solutions_data['y_ABM4'] = y_values
                exact_func = None
                if exact_sol_str:
                    try:
                        safe_dict = {"np": np}
                        exact_func = eval(f"lambda {indep_var}: {exact_sol_str}", safe_dict)
                        exact_func(np.array([t0, tend]))
                    except Exception as e:
                        st.warning(f"Lỗi khi phân tích giải pháp giải tích: {e}. Bỏ qua soảng.")
                        exact_func = None
                if t_values.size > 0:
                    df_results = create_results_dataframe(t_values, solutions_data, exact_func)
                    st.markdown("**Kết quả Tóm tắt và Phân tích Lỗi**")
                    st.dataframe(df_results.style.format("{:.6f}"))
                    fig = create_plot(df_results, f"So sánh Giải pháp cho y' = {func_str}")
                    st.pyplot(fig)
                else:
                    st.error("Bộ giải không trả về kết quả. (Nguyên nhân: ABM4 cần N >= 4 hoặc N quá nhỏ)")
            except Exception as e:
                st.error(f"Đã xảy ra lỗi trong quá trình chạy bộ giải: {e}")
                import traceback
                st.code(traceback.format_exc())
else:
    with main_col:
        st.info("Chào mừng! Vui lòng nhập các thông số của bạn vào biểu mẫu bên phải và nhấn 'Giải Phương trình'.")
        st.markdown("### Hướng dẫn sử dụng:")
        st.markdown("- **Bước 1:** Định nghĩa phương trình $y' = f(t,y)$ và các biến của bạn.")
        st.markdown("- **Bước 2:** Cung cấp điều kiện ban đầu ($t_0, y_0$), thời điểm kết thúc ($t_{end}$), và số bước ($N$).")
        st.markdown("- **Bước 3 (Tùy chọn):** Cung cấp giải pháp giải tích chính xác để so sánh lỗi. Sử dụng cú pháp `np.` (ví dụ: `np.exp(t)`).")
        st.markdown("- **Bước 4:** Chọn phương pháp số để giải.")
        st.markdown("- **Bước 5:** Nhấn 'Giải Phương trình' và xem kết quả.")