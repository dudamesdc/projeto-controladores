import streamlit as st
import sympy as sp
import numpy as np
import math
import control as ct
import matplotlib.pyplot as plt

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="Calculadora Universal - Sistemas de Controle", layout="wide")
st.title("Calculadora de Projeto de Controladores")


# --- FUNÇÕES MATEMÁTICAS E AUXILIARES ---
def calc_zeta(mp_percent):
    """Fórmula do Anexo 1 para fator de amortecimento"""
    mp = mp_percent / 100.0
    return np.sqrt((np.log(mp)**2) / (np.pi**2 + np.log(mp)**2))

def format_complex(c):
    return f"{c.real:.4f} {'+' if c.imag >= 0 else '-'} {abs(c.imag):.4f}j"

def sympy_to_tf(expr, s_sym):
    """Converte expressão do SymPy para TransferFunction da biblioteca Control"""
    expr_simp = sp.cancel(sp.simplify(expr))
    num, den = sp.fraction(expr_simp)
    num_coeffs = [float(sp.N(c)) for c in sp.Poly(num, s_sym).all_coeffs()]
    den_coeffs = [float(sp.N(c)) for c in sp.Poly(den, s_sym).all_coeffs()]
    return ct.TransferFunction(num_coeffs, den_coeffs)

# --- MENU DE TEMPLATES ---
st.sidebar.header("Dados do Projeto")
st.sidebar.markdown("Selecione uma questão base e altere os valores abaixo se necessário.")
modo = st.sidebar.selectbox(
    "Escolha o Molde:",
    [
        "Modo Livre",
        "Questão 1 (PD, Mp, ts)",
        "Questão 2 (PD, Zeta, Wn)",
        "Questão 3 (PI, Pólos Explicitos)",
        "Questão 4 (PID, Mp, ts)",
        "Questão 5 (PID + Tustin)",
        "Questão 6 (Compensador + Euler)"
    ]
)

# Valores padrão baseados no template
def_G, def_H = "1", "1"
def_req = "Overshoot (MP) e Tempo (ts)"
def_ctrl = "PD"
def_disc = "Nenhum"
def_mp, def_ts, def_zeta, def_wn, def_real, def_imag, def_T = 10.0, 4.0, 0.7, 0.5, -4.0, 4.0, 1.0

if modo == "Questão 1 (PD, Mp, ts)":
    def_G, def_H = "4*(s+4) / (s**3 + 4*s**2 + 4*s)", "1"
    def_req, def_ctrl = "Overshoot (MP) e Tempo (ts)", "PD"
    def_mp, def_ts = 10.0, 4.0
elif modo == "Questão 2 (PD, Zeta, Wn)":
    def_G, def_H = "1 / (10000*(s**2 - 1.1772))", "1"
    def_req, def_ctrl = "Zeta e Wn explícitos", "PD"
    def_zeta, def_wn = 0.7, 0.5
elif modo == "Questão 3 (PI, Pólos Explicitos)":
    def_G, def_H = "5*(s**2 + 5*s + 4) / (s**2 + 4*s + 4)", "0.2 / (s + 1)"
    def_req, def_ctrl = "Pólos explícitos", "PI"
    def_real, def_imag = -4.0, 4.0
elif modo == "Questão 4 (PID, Mp, ts)":
    def_G, def_H = "5 / (s**3 + 12*s**2 + 22*s + 20)", "0.5"
    def_req, def_ctrl = "Overshoot (MP) e Tempo (ts)", "PID (Zeros iguais)"
    def_mp, def_ts = 20.0, 5.0
elif modo == "Questão 5 (PID + Tustin)":
    def_G, def_H = "5*(s+3) / (s*(s+4))", "1 / (s+1)"
    def_req, def_ctrl, def_disc = "Overshoot (MP) e Tempo (ts)", "PID (Zeros iguais)", "Tustin (Bilinear)"
    def_mp, def_ts, def_T = 10.0, 3.0, 2.0
elif modo == "Questão 6 (Compensador + Euler)":
    def_G, def_H = "2*(s+1) / (s**2 + 2*s + 2)", "(s+3) / (s+5)"
    def_req, def_ctrl, def_disc = "Pólos explícitos", "Compensador ( a / (s+b) )", "Euler (s = (z-1)/T)"
    def_real, def_imag, def_T = -2.5, 2.0, 1.0

# --- INTERFACE DE ENTRADA (SIDEBAR) ---
st.sidebar.divider()
st.sidebar.header("1. Dinâmica do Sistema")
G_str = st.sidebar.text_input("G(s)", value=def_G)
H_str = st.sidebar.text_input("H(s)", value=def_H)

st.sidebar.header("2. Requisitos de Desempenho")
req_options = ["Overshoot (MP) e Tempo (ts)", "Zeta e Wn explícitos", "Pólos explícitos"]
req_type = st.sidebar.selectbox("Tipo de Requisito", req_options, index=req_options.index(def_req))

s_d = None
ts_crit = "5%"
if req_type == "Overshoot (MP) e Tempo (ts)":
    mp_val = st.sidebar.number_input("M_P máximo (%)", min_value=0.1, value=def_mp)
    ts_val = st.sidebar.number_input("Tempo de Acomodação (s)", min_value=0.1, value=def_ts)
    ts_crit = st.sidebar.radio("Critério de ts", ["5%", "2%"], horizontal=True)
elif req_type == "Zeta e Wn explícitos":
    zeta_val = st.sidebar.number_input("Fator de Amortecimento (Zeta)", min_value=0.01, max_value=0.99, value=def_zeta)
    wn_val = st.sidebar.number_input("Frequência Natural (Wn) rad/s", min_value=0.1, value=def_wn)
elif req_type == "Pólos explícitos":
    real_part = st.sidebar.number_input("Parte Real (Sigma)", value=def_real)
    imag_part = st.sidebar.number_input("Parte Imaginária (Wd)", min_value=0.0, value=def_imag)

st.sidebar.header("3. Tipo de Controlador")
ctrl_options = [
    "PD", 
    "PI", 
    "PID (Zeros iguais)", 
    "Compensador ( a / (s+b) )", 
    "Construtor Livre (Personalizado)"
]
ctrl_index = ctrl_options.index(def_ctrl) if def_ctrl in ctrl_options else 0
ctrl_type = st.sidebar.selectbox("Controlador", ctrl_options, index=ctrl_index)

if ctrl_type == "Construtor Livre (Personalizado)":
    st.sidebar.info("Digite a estrutura do Gc sem o ganho Kc.")
    Gc_custom_str = st.sidebar.text_input("Estrutura (Ex: (s+3)/(s+a))", "1 / (s + a)")
    var_custom = st.sidebar.text_input("Nome da variável desconhecida (Ex: a)", "a")

st.sidebar.header("4. Discretização")
disc_options = ["Nenhum", "Euler (s = (z-1)/T)", "Tustin (Bilinear)"]
disc_type = st.sidebar.selectbox("Método", disc_options, index=disc_options.index(def_disc))
T_val = st.sidebar.number_input("Período de Amostragem T (s)", min_value=0.01, value=def_T) if disc_type != "Nenhum" else None

st.sidebar.divider()
# --- BOTÃO DE CALCULAR ---
calcular = st.sidebar.button("Calcular Projeto", type="primary", use_container_width=True)

# --- PROCESSAMENTO PRINCIPAL ---
if calcular:
    s, z = sp.symbols('s z')

    try:
        # Parsing das expressões
        G_expr = sp.sympify(G_str)
        H_expr = sp.sympify(H_str)
        
        st.header("Passo 1: Definição dos Pólos Desejados")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Cálculos dos Parâmetros:**")
            if req_type == "Overshoot (MP) e Tempo (ts)":
                zeta = calc_zeta(mp_val)
                wn = (3.0 if ts_crit == "5%" else 4.0) / (zeta * ts_val)
                wd = wn * np.sqrt(1 - zeta**2)
                s_d = complex(-zeta * wn, wd)
                st.latex(r"\zeta = \sqrt{\frac{(\ln(MP/100))^2}{\pi^2 + (\ln(MP/100))^2}} \approx " + f"{zeta:.4f}")
                st.latex(r"\omega_n = \frac{" + ("3" if ts_crit == "5%" else "4") + r"}{\zeta t_s} \approx " + f"{wn:.4f}" + r" \text{ rad/s}")
            elif req_type == "Zeta e Wn explícitos":
                zeta = zeta_val
                wn = wn_val
                wd = wn * np.sqrt(1 - zeta**2)
                s_d = complex(-zeta * wn, wd)
                st.latex(r"\omega_d = \omega_n\sqrt{1-\zeta^2} \approx " + f"{wd:.4f}" + r" \text{ rad/s}")
            else:
                s_d = complex(real_part, imag_part)
                st.write("Pólos fornecidos explicitamente.")
        
        with col2:
            st.success(f"**Pólo Dominante de Malha Fechada ($s_d$):**\n\n$s_d = {format_complex(s_d)}$")

        st.divider()

        st.header("Passo 2: Critério do Ângulo (Soma das Fases = 180°)")
        GH_expr = G_expr * H_expr
        
        # Extração das raízes para exibir o cálculo detalhado dos ângulos
        GH_simp = sp.cancel(sp.simplify(GH_expr))
        num, den = sp.fraction(GH_simp)
        
        ang_GH = 0.0
        try:
            zeros_planta = sp.nroots(num)
            poles_planta = sp.nroots(den)
            
            st.markdown("**Contribuição angular da planta e realimentação no ponto $s_d$:**")
            
            # Calculando o ângulo da constante de ganho estático (se houver sinal negativo)
            # Geralmente é 0 para sistemas estáveis, mas é bom garantir matematicamente.
            GH_val_pure = complex(GH_expr.subs(s, s_d).evalf())
            
            for i, z_val in enumerate(zeros_planta):
                z_cplx = complex(z_val)
                ang = np.degrees(np.angle(s_d - z_cplx))
                ang_GH += ang
                st.latex(rf"\theta_{{z{i+1}}} = \angle(s_d - ({format_complex(z_cplx)})) = {ang:.2f}^\circ")
                
            for i, p_val in enumerate(poles_planta):
                p_cplx = complex(p_val)
                ang = np.degrees(np.angle(s_d - p_cplx))
                ang_GH -= ang
                st.latex(rf"\theta_{{p{i+1}}} = \angle(s_d - ({format_complex(p_cplx)})) = {ang:.2f}^\circ")
                
            # Correção final pelo valor absoluto substituído (para captar o ganho K natural da planta)
            ang_GH = np.degrees(np.angle(GH_val_pure))
            st.latex(rf"\phi = \sum \theta_z - \sum \theta_p = {ang_GH:.2f}^\circ")
            
        except Exception:
            # Fallback seguro caso a extração de raízes não funcione para funções não polinomiais
            GH_val_pure = complex(GH_expr.subs(s, s_d).evalf())
            ang_GH = np.degrees(np.angle(GH_val_pure))
            st.write(rf"Ângulo total da malha aberta não compensada $\phi = \angle G(s_d)H(s_d) = {ang_GH:.2f}^\circ$")

        ang_needed = 180 - ang_GH
        while ang_needed <= 0: ang_needed += 360
        while ang_needed > 180: ang_needed -= 360
        
        st.markdown("**Cálculo do Controlador:**")
        st.write(f"Para satisfazer o critério, o controlador precisa fornecer $\\theta_c = {ang_needed:.2f}^\\circ$")
        
        zc_val = None
        Gc_estrutura = None
        
        if ctrl_type == "PD":
            st.markdown("Estrutura do PD: $G_c(s) = K_c(s+z_c)$")
            zc_val = abs(s_d.imag) / math.tan(math.radians(ang_needed)) - s_d.real
            st.latex(rf"\tan({ang_needed:.2f}^\circ) = \frac{{\omega_d}}{{z_c - \sigma}} = \frac{{{abs(s_d.imag):.4f}}}{{z_c - ({s_d.real:.4f})}}")
            st.latex(rf"z_c = {zc_val:.4f} \implies G_{{c,estrut}}(s) = s + {zc_val:.4f}")
            Gc_estrutura = s + zc_val

        elif ctrl_type == "PI":
            st.markdown("Estrutura do PI: $G_c(s) = K_c\frac{(s+z_c)}{s}$")
            ang_polo_origem = np.degrees(np.angle(s_d))
            ang_zero_pi = ang_needed + ang_polo_origem
            while ang_zero_pi <= 0: ang_zero_pi += 360
            while ang_zero_pi > 180: ang_zero_pi -= 360
            st.write(f"O pólo na origem em $s=0$ contribui com $\\angle s_d = {ang_polo_origem:.2f}^\\circ$. Logo, o zero deve fornecer $\\theta_z = {ang_zero_pi:.2f}^\\circ$")
            zc_val = abs(s_d.imag) / math.tan(math.radians(ang_zero_pi)) - s_d.real
            st.latex(rf"\tan({ang_zero_pi:.2f}^\circ) = \frac{{{abs(s_d.imag):.4f}}}{{z_c - ({s_d.real:.4f})}} \implies z_c = {zc_val:.4f}")
            Gc_estrutura = (s + zc_val) / s

        elif ctrl_type == "PID (Zeros iguais)":
            st.markdown(r"Estrutura do PID: $G_c(s) = K_c\frac{(s+z_c)^2}{s}$")
            ang_polo_origem = np.degrees(np.angle(s_d))
            ang_zero_total = ang_needed + ang_polo_origem
            ang_cada_zero = ang_zero_total / 2.0
            while ang_cada_zero <= 0: ang_cada_zero += 180
            st.write(f"Pólo na origem: $\\angle s_d = {ang_polo_origem:.2f}^\\circ$. Contribuição exigida de CADA zero do PID: $\\theta_z = {ang_cada_zero:.2f}^\\circ$")
            zc_val = abs(s_d.imag) / math.tan(math.radians(ang_cada_zero)) - s_d.real
            st.latex(rf"\tan({ang_cada_zero:.2f}^\circ) = \frac{{{abs(s_d.imag):.4f}}}{{z_c - ({s_d.real:.4f})}} \implies z_c = {zc_val:.4f}")
            Gc_estrutura = ((s + zc_val)**2) / s

        elif ctrl_type == "Compensador ( a / (s+b) )":
            st.markdown(r"Estrutura do Compensador: $G_c(s) = \frac{a}{s+b}$")
            ang_polo = ang_GH - 180
            while ang_polo <= 0: ang_polo += 360
            while ang_polo > 180: ang_polo -= 360
            b_val = abs(s_d.imag) / math.tan(math.radians(ang_polo)) - s_d.real
            st.write(f"Neste caso, o compensador tem um pólo ($-b$). Ele deve subtrair fase para fechar $180^\circ$. $\\angle(s_d+b) = {ang_polo:.2f}^\circ$")
            st.latex(rf"\tan({ang_polo:.2f}^\circ) = \frac{{{abs(s_d.imag):.4f}}}{{b - ({s_d.real:.4f})}} \implies b = {b_val:.4f}")
            Gc_estrutura = 1 / (s + b_val)

        elif ctrl_type == "Construtor Livre (Personalizado)":
            var_sym = sp.Symbol(var_custom)
            Gc_expr_custom = sp.sympify(Gc_custom_str)
            st.markdown(f"**Estrutura do Compensador:** $G_{{c,estrut}}(s) =$ " + f"${sp.latex(Gc_expr_custom)}$")
            target_rad = math.radians(ang_needed)
            target_vec = complex(math.cos(target_rad), math.sin(target_rad))
            
            def eval_angle_diff(val):
                try:
                    c_val = complex(Gc_expr_custom.subs({s: s_d, var_sym: val}).evalf())
                    return (c_val * target_vec.conjugate()).imag
                except:
                    return 1e9

            x0, x1 = 0.0, 1.0 
            for _ in range(100):
                f0 = eval_angle_diff(x0)
                f1 = eval_angle_diff(x1)
                if abs(f1) < 1e-6: break
                if f1 - f0 == 0: x1 += 0.1; f1 = eval_angle_diff(x1)
                x_next = x1 - f1 * (x1 - x0) / (f1 - f0)
                x0, x1 = x1, x_next
                
            solved_var = x1
            st.latex(f"{var_custom} = {solved_var:.4f}")
            Gc_estrutura = Gc_expr_custom.subs(var_sym, solved_var)

        st.divider()

        st.header("Passo 3: Critério do Módulo (Encontrar Ganho)")
        st.markdown("O módulo da função de malha aberta avaliada em $s_d$ deve ser igual a 1.")
        st.latex(r"|G_c(s_d)G(s_d)H(s_d)| = 1 \implies K_c = \frac{1}{|G_{c,estrut}(s_d)| \cdot |G(s_d)H(s_d)|}")
        
        try:
            st.markdown("**Distâncias (Módulos) calculadas do ponto $s_d$ até as raízes da planta:**")
            for i, z_val in enumerate(zeros_planta):
                dist = abs(s_d - complex(z_val))
                st.latex(rf"|s_d - z_{i+1}| = |s_d - ({format_complex(complex(z_val))})| = {dist:.4f}")
            for i, p_val in enumerate(poles_planta):
                dist = abs(s_d - complex(p_val))
                st.latex(rf"|s_d - p_{i+1}| = |s_d - ({format_complex(complex(p_val))})| = {dist:.4f}")
        except Exception:
            pass # Ignora caso a planta seja não-polinomial
        
        mag_Gc_est = abs(complex(Gc_estrutura.subs(s, s_d).evalf()))
        mag_GH = abs(complex(GH_expr.subs(s, s_d).evalf()))
        
        st.write("Substituindo os módulos parciais:")
        st.latex(rf"|G_{{c,estrut}}(s_d)| = {mag_Gc_est:.4f}")
        st.latex(rf"|G(s_d)H(s_d)| = {mag_GH:.4f}")
        
        Kc_val = 1.0 / (mag_Gc_est * mag_GH)
        st.latex(rf"K_c = \frac{{1}}{{{mag_Gc_est:.4f} \cdot {mag_GH:.4f}}} = {Kc_val:.4f}")
        
        # Manter a estrutura real para os cálculos matemáticos do Passo 4 e 5
        Gc_final = Kc_val * Gc_estrutura 
        
        # Criação de um espelho visual "Proibindo" o chuveirinho para ficar visualmente correto na tela
        Gc_display = sp.Mul(round(Kc_val, 4), sp.N(Gc_estrutura, 4), evaluate=False)
        
        st.success(f"**Função de Transferência do Controlador Contínuo:**")
        st.latex(r"G_c(s) = " + sp.latex(Gc_display))

        st.divider()

        if disc_type != "Nenhum":
            st.header(f"Passo 4: Discretização ({disc_type})")
            if "Euler" in disc_type:
                s_sub = (z - 1) / T_val
                st.latex(r"s = \frac{z-1}{T}")
            else:
                s_sub = (2 / T_val) * ((z - 1) / (z + 1))
                st.latex(r"s = \frac{2}{T}\frac{z-1}{z+1}")
                
            Gc_z = Gc_final.subs(s, s_sub).simplify()
            st.success(f"**Controlador Discreto $G_c(z)$ com $T = {T_val}s$:**")
            st.latex(r"G_c(z) = " + sp.latex(sp.N(Gc_z, 4)))

        st.divider()

        # --- NOVA SEÇÃO: PASSO 5 - SIMULAÇÃO ---
        st.header("Passo 5: Simulação da Resposta no Tempo")
        st.markdown("Verificação do comportamento dinâmico (Ajuste Fino)")
        
        with st.spinner("Compilando as funções de transferência e simulando..."):
            sys_G = sympy_to_tf(G_expr, s)
            sys_H = sympy_to_tf(H_expr, s)
            sys_Gc = sympy_to_tf(Gc_final, s)

            # Malhas Fechadas
            sys_MF_original = ct.feedback(sys_G, sys_H)
            sys_MF_controlado = ct.feedback(sys_Gc * sys_G, sys_H)

            # Determinar o tempo de simulação ideal com base no sistema controlado
            info = ct.step_info(sys_MF_controlado)
            ts_real = info['SettlingTime']
            
            # Garantir que a simulação vá um pouco além da acomodação
            t_max = ts_real * 2 if not np.isnan(ts_real) else 10.0
            t_sim = np.linspace(0, t_max, 1000)

            # Simulação do degrau
            t_orig, y_orig = ct.step_response(sys_MF_original, T=t_sim)
            t_cont, y_cont = ct.step_response(sys_MF_controlado, T=t_sim)

            # Extração do M_P real para exibir no gráfico
            mp_real = info['Overshoot']
            tp_real = info['PeakTime']

            # Plotagem idêntica à do Anexo
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(t_orig, y_orig, 'r-', linewidth=2, label="Sem Controlador")
            ax.plot(t_cont, y_cont, 'b-', linewidth=2, label="Com Controlador")
            
            # Linhas de referência (1.0 e bandas de acomodação 2% ou 5%)
            ax.axhline(1, color='k', linestyle='-', linewidth=1)
            erro_ts = 0.05 if ts_crit == "5%" else 0.02
            ax.axhline(1 + erro_ts, color='c', linestyle='--', linewidth=1, alpha=0.7)
            ax.axhline(1 - erro_ts, color='c', linestyle='--', linewidth=1, alpha=0.7)
            
            # Anotar o pico
            ax.plot(tp_real, 1 + mp_real/100, 'ro')
            ax.text(tp_real + 0.1, 1 + mp_real/100, f"$M_P$ = {mp_real:.2f}%", color='blue', fontsize=12)
            
            # Anotar o tempo de acomodação
            ax.axvline(ts_real, color='r', linestyle='--', linewidth=1, alpha=0.7)
            ax.text(ts_real + 0.1, 0.8, f"$t_s$ = {ts_real:.2f}s", color='black', fontsize=12)

            ax.set_title("Resposta ao Degrau", fontsize=14)
            ax.set_xlabel("Tempo", fontsize=12)
            ax.set_ylabel("Resposta", fontsize=12)
            ax.legend(loc='lower right')
            ax.grid(True)
            
            st.pyplot(fig)
            
            st.info(f"**Desempenho Real (Simulado):** $M_P$ = {mp_real:.2f}%, $t_s$ = {ts_real:.2f}s.")

    except Exception as e:
        st.error("Erro ao processar as equações. Certifique-se de usar a sintaxe correta do Python (ex: `s**2` para potência, `*` para multiplicação).")
        st.exception(e)
else:
    st.info("Insira os parâmetros na barra lateral e clique em **Calcular Projeto** para iniciar.")