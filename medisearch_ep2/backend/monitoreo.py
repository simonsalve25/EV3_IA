import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(page_title="Dashboard de Observabilidad - Duoc UC", layout="wide")

st.title(" Dashboard de Observabilidad y Trazabilidad")
st.caption("Métricas en Tiempo Real del Agente de Inteligencia Artificial - ISY0101")

METRICAS_FILE = 'metricas_historicas.json'
LOGS_FILE = 'agente_observabilidad.log'

# Cargar Datos
if os.path.exists(METRICAS_FILE):
    with open(METRICAS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
else:
    df = pd.DataFrame()

if not df.empty:
    # 1. Indicadores clave (KPIs) - IE1, IE2
    col1, col2, col3, col4 = st.columns(4)
    
    df_exitosos = df[df['exito'] == True]
    total_peticiones = len(df)
    errores = len(df[df['exito'] == False])
    tasa_error = (errores / total_peticiones) * 100 if total_peticiones > 0 else 0
    
    latencia_promedio = df_exitosos['latencia'].mean() if not df_exitosos.empty else 0
    precision_promedio = df_exitosos['precision'].mean() if not df_exitosos.empty else 0
    
    col1.metric("Latencia Promedio", f"{latencia_promedio:.2f} s")
    col2.metric("Precisión Promedio", f"{precision_promedio:.1f} %")
    col3.metric("Tasa de Errores", f"{tasa_error:.1f} %", f"{errores} fallos", delta_color="inverse")
    col4.metric("Total Solicitudes", total_peticiones)
    
    st.markdown("---")
    
    # 2. Gráficos de Comportamiento Temporal - IE1, IE2, IE4
    graf1, graf2 = st.columns(2)
    
    with graf1:
        st.subheader("Rendimiento de Respuestas (Latencia y Consistencia)")
        st.line_chart(df, x="timestamp", y=["latencia", "consistencia"])
        
    with graf2:
        st.subheader("Consumo de Infraestructura (Sostenibilidad)")
        st.area_chart(df, x="timestamp", y=["cpu", "memoria"])
        
    # 3. Datos tabulares (IE4)
    st.subheader("Registro Histórico Detallado")
    st.dataframe(df.tail(10), use_container_width=True)

else:
    st.info("Esperando datos de telemetría. Interactúa con el agente desde tu interfaz web para poblar el Dashboard.")

# 4. Sección de Auditoría de Logs e Identificación de fallas (IE3, IE4)
st.markdown("---")
st.subheader("Trazabilidad de Logs en Vivo (`agente_observabilidad.log`)")
if os.path.exists(LOGS_FILE):
    with open(LOGS_FILE, 'r', encoding='utf-8') as f:
        log_lines = f.readlines()
    # Mostrar últimos 15 eventos del log de ejecución
    st.code("".join(log_lines[-20:]), language="log")
else:
    st.warning("No se ha detectado archivo de logs todavía.")
