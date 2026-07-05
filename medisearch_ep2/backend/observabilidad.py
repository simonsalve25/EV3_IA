import logging
import time
import psutil
import json
import os
from datetime import datetime

# Configuración de Logs del Sistema (Trazabilidad - IE3)
logging.basicConfig(
    filename='agente_observabilidad.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

METRICAS_FILE = 'metricas_historicas.json'

def registrar_ejecucion(prompt, exito, latencia, precision_estimada, consistencia, error_msg=""):
    """
    Registra métricas clave de rendimiento y uso de hardware (IE1, IE2, IE3).
    """
    # Captura de recursos de hardware actuales
    cpu_uso = psutil.cpu_percent()
    memoria_uso = psutil.virtual_memory().percent
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Mensaje formateado para el archivo log clásico (Trazabilidad)
    estado_str = "EXITO" if exito else f"ERROR: {error_msg}"
    log_line = f"Prompt: '{prompt}' | Estado: {estado_str} | Latencia: {latencia:.2f}s | CPU: {cpu_uso}% | RAM: {memoria_uso}%"
    
    if exito:
        if consistencia < 80:
            logging.warning(f"[CONSISTENCIA BAJA] {log_line}")
        else:
            logging.info(f"[PROCESADO] {log_line}")
    else:
        logging.error(f"[FALLO CRÍTICO] {log_line}")

    # Guardar en JSON estructurado para el Dashboard
    nuevo_registro = {
        "timestamp": timestamp,
        "prompt": prompt,
        "exito": exito,
        "latencia": latencia,
        "precision": precision_estimada if exito else 0.0,
        "consistencia": consistencia if exito else 0.0,
        "cpu": cpu_uso,
        "memoria": memoria_uso,
        "error": error_msg
    }
    
    registros = []
    if os.path.exists(METRICAS_FILE):
        try:
            with open(METRICAS_FILE, 'r', encoding='utf-8') as f:
                registros = json.load(f)
        except Exception:
            registros = []
            
    registros.append(nuevo_registro)
    
    with open(METRICAS_FILE, 'w', encoding='utf-8') as f:
        json.dump(registros, f, indent=4, ensure_ascii=False)
