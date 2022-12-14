# Pasos para ejecutar en la Raspberry PI 3

### Instalaciones necesarias
Para actualizar o instalar python

```
sudo apt install -y python3.10
```

Para verificar que haya quedado con la versión 3.10
```
python --version
```
Luego, actualizar pip
```
sudo -H pip3 install --upgrade pip
```

Luego, instalar python3.10-venv para poder crear un entorno virtual
```
sudo apt-get install python3.10-venv
```

Después crear el entorno virtual
```
python3.10 -m venv myprojectenv
```

Activar el entorno virtual
```
source myprojectenv/bin/activate
```

### Instalando librerias de Python
Uso el siguiente comando debido a la poca memoria que tiene la Raspberry, entonces no quiero cache
```
pip --no-cache-dir install tensorflow
```

Antes de instalar OpenCV instalo una dependencia que no tenía
```
sudo apt-get install libgl1
```

Después instalo OpenCV para capturar las imágenes de la cámara
```
pip --no-cache-dir install opencv-python
```
