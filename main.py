from PIL import Image
from ultralytics import YOLO  # Necesita instalarse con `pip install ultralytics`

def detectar_producto(ruta_imagen, modelo_pesos="yolov5s"):
    try:
        # Cargamos un modelo YOLO preentrenado
        modelo = YOLO(modelo_pesos)

        # Detectamos objetos en la imagen
        resultados = modelo.predict(ruta_imagen, conf=0.5)

        # Usamos el primer objeto detectado (por simplicidad, asumiendo que el catálogo tiene un producto principal)
        # Obtenemos las coordenadas del bounding box en formato (x_min, y_min, x_max, y_max)
        if resultados[0].boxes.shape[0] > 0:  # Verificamos si hubo detecciones
            x_min, y_min, x_max, y_max = resultados[0].boxes[0].xyxy[0].tolist()
            return int(x_min), int(y_min), int(x_max), int(y_max)
        else:
            raise ValueError("❌ No se detectó ningún producto en el catálogo.")
    except Exception as e:
        print(f"Error detectando producto: {str(e)}")
        return None


def crear_post_instagram(ruta_catalogo, ruta_encabezado, nombre_archivo, modelo_pesos="yolov5s"):
    try:
        # 1. Cargamos el catálogo y el encabezado de Expert
        catalogo = Image.open(ruta_catalogo)
        encabezado = Image.open(ruta_encabezado)
        
        # 2. Detectamos automáticamente las coordenadas del producto
        coordenadas = detectar_producto(ruta_catalogo, modelo_pesos)
        if not coordenadas:
            return  # Si no se detecta producto, detenemos el proceso
        
        # 3. Recortamos el producto
        producto = catalogo.crop(coordenadas)
        
        # 4. Creamos el fondo naranja (Color Expert #F36921)
        fondo = Image.new('RGB', (1080, 1080), (243, 105, 33))
        
        # 5. Pegamos el encabezado arriba (ajustando tamaño si es necesario)
        encabezado = encabezado.resize((1080, encabezado.size[1]), Image.Resampling.LANCZOS)
        fondo.paste(encabezado, (0, 0))
        
        # 6. Redimensionamos el producto para que ocupe máximo 800px de ancho
        ancho_max = 900
        ratio = ancho_max / float(producto.size[0])
        alto_nuevo = int(float(producto.size[1]) * ratio)
        producto = producto.resize((ancho_max, alto_nuevo), Image.Resampling.LANCZOS)
        
        # 7. Pegamos el producto en el centro del espacio sobrante
        pos_y = 350  # Altura debajo del encabezado 
        pos_x = (1080 - ancho_max) // 2
        fondo.paste(producto, (pos_x, pos_y))
        
        # 8. Guardamos el resultado
        fondo.save(f"{nombre_archivo}.jpg", "JPEG", quality=95)
        print(f"✅ Imagen {nombre_archivo} creada con éxito.")
        
    except FileNotFoundError as fnfe:
        print(f"❌ Error: No se pudo cargar una de las imágenes. Detalle: {fnfe}")
    except Exception as e:
        print(f"❌ Ha ocurrido un error inesperado: {str(e)}")


# --- EJEMPLO DE USO ---
crear_post_instagram(
    "catalogo.png", 
    "IMG_1676.jpeg", 
    "post_tv_tcl", 
    modelo_pesos="yolov5s"  # Cambia el modelo si tienes uno más específico (por ejemplo, entrenado con datos de catálogos).
)
