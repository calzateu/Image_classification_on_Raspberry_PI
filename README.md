# Pasos para ejecutar en la Raspberry PI 3

### Creación entorno virtual
Crear el entorno virtual
```
python3 -m venv myprojectenv
```

Activar el entorno virtual
```
source myprojectenv/bin/activate
```

### Instalando librerias de Python
Antes de instalar OpenCV instalo una dependencia que no tenía
```
sudo apt-get install libgl1
```

Para instalar opencv uso el siguiente comando debido a la poca memoria que tiene la Raspberry, entonces no quiero cache.
```
pip --no-cache-dir install opencv-python
```

Luego requests
```
pip install requests
```


