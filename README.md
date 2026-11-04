# Proyecto Pokémon: mapas de Kohonen

**Sofía Muñiz Garzón (9400632)**  
**Alberto Caballero Lorente (9003606)**

Repositorio: [Pokemon-Kohonen](https://github.com/sofiamgarz/Pokemon-Kohonen).

Implementación de un SOM en Python para analizar 1.025 especies de PokéAPI.
El modelo principal utiliza seis estadísticas base, altura y peso. Se incluyen
la comparación con K-means, la sensibilidad a la rejilla y las semillas y un
experimento que añade los tipos como variables de entrada.

## Ejecución

Se utilizó Python 3.12.14. `INSTRUCCIONES.txt` contiene los pasos para Windows,
macOS y Linux. Con el entorno activado, desde la raíz del repositorio:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/run.py
python src/build_report.py
```

Los CSV se incluyen en `data/raw` y se comprueban con el manifiesto
`data/procedencia.json`. El análisis funciona sin conexión después de instalar
las dependencias. Las seis pruebas comprueban el ejemplo manual, la actualización,
la vecindad, el decaimiento, la repetición del ajuste y el preprocesamiento.

## Archivos

- `src/data.py`: carga, comprobación, limpieza y escalado.
- `src/som.py`: algoritmo de Kohonen y medidas del mapa.
- `src/run.py`: experimentos y resultados.
- `src/plots.py`: visualizaciones.
- `src/build_report.py`: tablas con los resultados de los experimentos.
- `src/download_data.py`: recuperación opcional de los CSV de la misma versión.
- `tests/test_som.py`: comprobaciones del algoritmo y de los datos.
- `requirements.txt`: versiones utilizadas de las bibliotecas.

`run.py` genera `resultados/`, con las métricas, las asignaciones de especies,
los modelos y las figuras. `build_report.py` utiliza esos resultados para
crear las tablas en `resultados/tablas/`.
