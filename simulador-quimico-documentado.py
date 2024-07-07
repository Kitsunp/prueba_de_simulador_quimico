# Importaciones
import numpy as np  # Para cálculos numéricos eficientes
import matplotlib.pyplot as plt  # Para visualización de datos
from mpl_toolkits.mplot3d import Axes3D  # Para gráficos 3D
from scipy.spatial import distance_matrix  # Para calcular distancias entre partículas
from scipy.constants import k, epsilon_0, e  # Constantes físicas (Boltzmann, permitividad del vacío, carga elemental)
from scipy.linalg import eigh  # Para cálculos de eigenvalores y eigenvectores
from matplotlib.colors import LinearSegmentedColormap  # Para mapas de colores personalizados
from mpl_toolkits.mplot3d import proj3d  # Para proyecciones 3D
from matplotlib.patches import FancyArrowPatch  # Para dibujar flechas en gráficos

# Clase Arrow3D
class Arrow3D(FancyArrowPatch):
    """
    Clase para crear flechas en gráficos 3D.
    Hereda de FancyArrowPatch y sobrescribe métodos para funcionar en 3D.
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        """
        Inicializa la flecha 3D.
        :param xs, ys, zs: Coordenadas de inicio y fin de la flecha
        """
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        """
        Dibuja la flecha en el renderizador.
        :param renderer: Renderizador de matplotlib
        """
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        """
        Realiza la proyección 3D de la flecha.
        :param renderer: Renderizador (opcional)
        :return: Profundidad mínima de la flecha proyectada
        """
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)

# Clase principal SimuladorQuimico
class SimuladorQuimico:
    """
    Clase principal para simular estructuras químicas y sus propiedades.
    """
    def __init__(self, num_particulas=50, tipo_red='FCC'):
        """
        Inicializa el simulador químico.
        :param num_particulas: Número de partículas en la simulación
        :param tipo_red: Tipo de red cristalina ('FCC', 'BCC', o aleatorio si no se especifica)
        """
        self.num_particulas = num_particulas
        self.tipo_red = tipo_red
        self.generar_estructura_cristalina()
        self.inicializar_propiedades_particulas()
        self.calcular_propiedades_material()
        self.calcular_enlaces()
        self.calcular_hamiltoniano()

    def generar_estructura_cristalina(self):
        """
        Genera la estructura cristalina basada en el tipo de red especificado.
        Soporta redes FCC (cúbica centrada en las caras), BCC (cúbica centrada en el cuerpo),
        o una distribución aleatoria si no se especifica un tipo válido.
        """
        if self.tipo_red == 'FCC':
            # Genera una red FCC
            lado = int(np.ceil((self.num_particulas/4)**(1/3)))
            x, y, z = np.meshgrid(range(lado), range(lado), range(lado))
            posiciones = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
            posiciones_fcc = np.vstack([posiciones, 
                                        posiciones + [0.5, 0.5, 0],
                                        posiciones + [0.5, 0, 0.5],
                                        posiciones + [0, 0.5, 0.5]])
            self.posiciones = posiciones_fcc[:self.num_particulas]
        elif self.tipo_red == 'BCC':
            # Genera una red BCC
            lado = int(np.ceil((self.num_particulas/2)**(1/3)))
            x, y, z = np.meshgrid(range(lado), range(lado), range(lado))
            posiciones = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
            posiciones_bcc = np.vstack([posiciones, posiciones + [0.5, 0.5, 0.5]])
            self.posiciones = posiciones_bcc[:self.num_particulas]
        else:
            # Genera posiciones aleatorias si no se especifica un tipo de red válido
            self.posiciones = np.random.rand(self.num_particulas, 3) * 3

        # Aplica condiciones de contorno periódicas
        self.caja = np.max(self.posiciones, axis=0)
        self.posiciones = self.posiciones % self.caja

    def inicializar_propiedades_particulas(self):
        """
        Inicializa las propiedades de las partículas de manera realista.
        Asigna elementos, masas, radios atómicos, electronegatividades y cargas.
        """
        # Define propiedades para algunos elementos comunes
        elementos = ['H', 'C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Fe', 'Cu', 'Ag', 'Au']
        masas = [1.008, 12.01, 14.01, 16.00, 22.99, 24.31, 26.98, 28.09, 30.97, 32.07, 35.45, 39.10, 40.08, 55.85, 63.55, 107.87, 196.97]
        radios = [0.53, 0.67, 0.56, 0.48, 1.90, 1.45, 1.18, 1.11, 0.98, 0.88, 0.79, 2.43, 1.94, 1.24, 1.28, 1.45, 1.37]
        electronegatividades = [2.20, 2.55, 3.04, 3.44, 0.93, 1.31, 1.61, 1.90, 2.19, 2.58, 3.16, 0.82, 1.00, 1.83, 1.90, 1.93, 2.54]

        # Asigna propiedades aleatorias a cada partícula
        indices = np.random.choice(len(elementos), self.num_particulas)
        self.elementos = np.array(elementos)[indices]
        self.masas = np.array(masas)[indices]
        self.radios_atomicos = np.array(radios)[indices]
        self.electronegatividades = np.array(electronegatividades)[indices]
        
        # Calcula cargas basadas en diferencias de electronegatividad
        self.cargas = np.zeros(self.num_particulas)
        for i in range(self.num_particulas):
            for j in range(i+1, self.num_particulas):
                diff_electroneg = self.electronegatividades[j] - self.electronegatividades[i]
                carga = 0.16 * diff_electroneg + 0.035 * diff_electroneg**2
                self.cargas[i] -= carga
                self.cargas[j] += carga
        
        # Convierte cargas a unidades fundamentales
        self.cargas *= e

    def calcular_propiedades_material(self):
        """
        Calcula propiedades macroscópicas del material de manera aproximada.
        Incluye volumen, densidad, temperatura, módulo de Young, conductividad térmica y temperatura de fusión.
        """
        self.volumen = np.prod(self.caja)
        self.densidad = np.sum(self.masas) / self.volumen
        self.temperatura = 300  # Kelvin, temperatura ambiente por defecto

        # Módulo de Young (aproximación simplificada)
        self.modulo_young = np.mean(self.electronegatividades) * 50  # GPa

        # Conductividad térmica (aproximación simplificada)
        self.conductividad_termica = np.mean(self.electronegatividades) * 20  # W/(m·K)

        # Temperatura de fusión (aproximación simplificada)
        self.temperatura_fusion = np.mean(self.electronegatividades) * 500  # K

    def calcular_enlaces(self):
        """
        Calcula los enlaces entre partículas basados en la distancia.
        Utiliza una matriz de distancias y un umbral basado en el radio atómico promedio.
        """
        dist_matriz = distance_matrix(self.posiciones, self.posiciones)
        np.fill_diagonal(dist_matriz, np.inf)
        self.enlaces = dist_matriz < np.mean(self.radios_atomicos) * 2.5

    def calcular_hamiltoniano(self):
        """
        Calcula el Hamiltoniano del sistema.
        Incluye interacciones coulombianas entre partículas y energía cinética.
        """
        self.H = np.zeros((self.num_particulas, self.num_particulas))
        for i in range(self.num_particulas):
            for j in range(self.num_particulas):
                if i != j:
                    r = np.linalg.norm(self.posiciones[i] - self.posiciones[j])
                    if r > 1e-10:  # Evita división por cero
                        self.H[i, j] = self.cargas[i] * self.cargas[j] / (4 * np.pi * epsilon_0 * r)
                    else:
                        self.H[i, j] = 0  # Establece a cero si las partículas están demasiado cerca
        
        # Añade energía cinética en la diagonal
        for i in range(self.num_particulas):
            self.H[i, i] = 3 * k * self.temperatura / 2

    def calcular_energia_total(self):
        """
        Calcula la energía total del sistema.
        Utiliza los eigenvalores del Hamiltoniano.
        :return: Energía total del sistema
        """
        eigenvalores, _ = eigh(self.H)
        return np.sum(eigenvalores[:self.num_particulas])  # Supone que todas las partículas están en el estado fundamental

    def simular_dinamica_molecular(self, pasos=1000, dt=1e-15):
        """
        Realiza una simulación de dinámica molecular.
        :param pasos: Número de pasos de simulación
        :param dt: Paso de tiempo en segundos
        :return: Listas de temperaturas y energías durante la simulación
        """
        velocidades = np.random.randn(self.num_particulas, 3) * np.sqrt(k * self.temperatura / self.masas[:, np.newaxis])
        temperaturas = []
        energias = []

        for _ in range(pasos):
            fuerzas = self.calcular_fuerzas()
            velocidades += fuerzas * dt / self.masas[:, np.newaxis]
            self.posiciones += velocidades * dt
            
            # Aplica condiciones de contorno periódicas
            self.posiciones %= self.caja

            energia_cinetica = 0.5 * np.sum(self.masas[:, np.newaxis] * velocidades**2)
            energia_potencial = self.calcular_energia_potencial()
            energias.append(energia_cinetica + energia_potencial)
            
            temperatura = 2 * energia_cinetica / (3 * self.num_particulas * k)
            temperaturas.append(temperatura)

            # Aplica termostato de Berendsen
            factor_escala = np.sqrt(1 + (dt / 100) * (self.temperatura / temperatura - 1))
            velocidades *= factor_escala

        return temperaturas, energias

    def calcular_fuerzas(self):
        """
        Calcula las fuerzas entre partículas.
        Incluye fuerzas de Coulomb y Lennard-Jones.
        :return: Array de fuerzas para cada partícula
        """
        fuerzas = np.zeros_like(self.posiciones)
        for i in range(self.num_particulas):
            for j in range(i+1, self.num_particulas):
                r = self.posiciones[j] - self.posiciones[i]
                r = r - np.round(r / self.caja) * self.caja  # Mínima imagen
                r_mag = np.linalg.norm(r)
                
                if r_mag > 1e-10:  # Evita división por cero
                    r_unit = r / r_mag
                    
                    # Fuerza de Coulomb
                    fuerza_coulomb = self.cargas[i] * self.cargas[j] / (4 * np.pi * epsilon_0 * r_mag**2) * r_unit
                    
                    # Fuerza de Lennard-Jones
                    sigma = (self.radios_atomicos[i] + self.radios_atomicos[j]) / 2
                    epsilon = np.sqrt(self.electronegatividades[i] * self.electronegatividades[j]) * 0.1  # Aproximación
                    fuerza_lj = 4 * epsilon * (12 * (sigma/r_mag)**13 - 6 * (sigma/r_mag)**7) / r_mag * r_unit
                    
                    fuerza_total = fuerza_coulomb + fuerza_lj
                    fuerzas[i] += fuerza_total
                    fuerzas[j] -= fuerza_total

        return fuerzas
    def calcular_energia_potencial(self):
        """
        Calcula la energía potencial del sistema.
        Incluye energía de Coulomb y Lennard-Jones.
        :return: Energía potencial total del sistema
        """
        energia = 0  # Inicializa la energía total a cero
        for i in range(self.num_particulas):
            for j in range(i+1, self.num_particulas):
                # Calcula el vector de distancia entre partículas
                r = self.posiciones[j] - self.posiciones[i]
                # Aplica condiciones de contorno periódicas (mínima imagen)
                r = r - np.round(r / self.caja) * self.caja
                # Calcula la magnitud de la distancia
                r_mag = np.linalg.norm(r)
                
                # Calcula la energía de Coulomb
                energia_coulomb = self.cargas[i] * self.cargas[j] / (4 * np.pi * epsilon_0 * r_mag)
                
                # Calcula los parámetros para la energía de Lennard-Jones
                sigma = (self.radios_atomicos[i] + self.radios_atomicos[j]) / 2
                epsilon = np.sqrt(self.electronegatividades[i] * self.electronegatividades[j]) * 0.1  # Aproximación
                # Calcula la energía de Lennard-Jones
                energia_lj = 4 * epsilon * ((sigma/r_mag)**12 - (sigma/r_mag)**6)
                
                # Suma las energías al total
                energia += energia_coulomb + energia_lj
        return energia

    def representacion_3d(self):
        """
        Crea una representación 3D de la estructura cristalina.
        Muestra las partículas como esferas coloreadas según su carga y los enlaces entre ellas.
        """
        fig = plt.figure(figsize=(12, 10))  # Crea una nueva figura
        ax = fig.add_subplot(111, projection='3d')  # Añade un subplot 3D
        
        # Normaliza las cargas para el mapa de colores
        norm = plt.Normalize(vmin=min(self.cargas), vmax=max(self.cargas))
        cmap = plt.get_cmap('viridis')  # Elige el mapa de colores
        
        # Dibuja las partículas
        scatter = ax.scatter(self.posiciones[:, 0], self.posiciones[:, 1], self.posiciones[:, 2],
                             s=self.radios_atomicos*500, c=self.cargas, cmap=cmap, alpha=0.7)
        
        # Añade etiquetas de elementos a las partículas
        for i, (pos, elemento) in enumerate(zip(self.posiciones, self.elementos)):
            ax.text(*pos, elemento, fontsize=8)
        
        # Dibuja los enlaces entre partículas
        for i in range(self.num_particulas):
            for j in range(i+1, self.num_particulas):
                if self.enlaces[i, j]:
                    start = self.posiciones[i]
                    end = self.posiciones[j]
                    ax.plot(*zip(start, end), color='gray', alpha=0.5)
        
        # Configura los ejes y el título
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Estructura del Material ({self.tipo_red})')
        
        # Añade una barra de color
        cbar = fig.colorbar(scatter, ax=ax, label='Carga (e)')
        cbar.set_label('Carga (e)', rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.show()

    def representacion_molecular_3d(self):
        """
        Crea una representación molecular 3D más detallada.
        Muestra las partículas como esferas coloreadas según su electronegatividad y los enlaces como flechas.
        """
        fig = plt.figure(figsize=(12, 10))  # Crea una nueva figura
        ax = fig.add_subplot(111, projection='3d')  # Añade un subplot 3D
        
        # Normaliza las electronegatividades para el mapa de colores
        norm = plt.Normalize(vmin=min(self.electronegatividades), vmax=max(self.electronegatividades))
        cmap = plt.get_cmap('coolwarm')  # Elige el mapa de colores
        
        # Dibuja las partículas
        scatter = ax.scatter(self.posiciones[:, 0], self.posiciones[:, 1], self.posiciones[:, 2],
                             s=self.radios_atomicos*1000, c=self.electronegatividades, cmap=cmap, alpha=0.7, edgecolors='k')
        
        # Añade etiquetas de elementos a las partículas
        for i, (pos, elemento) in enumerate(zip(self.posiciones, self.elementos)):
            ax.text(*pos, elemento, fontsize=8, ha='center', va='center')
        
        # Dibuja los enlaces entre partículas como flechas
        for i in range(self.num_particulas):
            for j in range(i+1, self.num_particulas):
                if self.enlaces[i, j]:
                    start = self.posiciones[i]
                    end = self.posiciones[j]
                    if np.all(np.isfinite(start)) and np.all(np.isfinite(end)):
                        arrow = Arrow3D([start[0], end[0]], [start[1], end[1]], 
                                        [start[2], end[2]], mutation_scale=20, 
                                        lw=2, arrowstyle="-", color="gray")
                        ax.add_artist(arrow)
        
        # Configura los ejes y el título
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Representación Molecular 3D')
        
        # Añade una barra de color
        cbar = fig.colorbar(scatter, ax=ax, label='Electronegatividad')
        cbar.set_label('Electronegatividad', rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.show()

    def representacion_macroscopica(self):
        """
        Crea una representación gráfica de las propiedades macroscópicas del material.
        Muestra cuatro gráficos: distribución de elementos, masa vs carga, distribución de electronegatividades,
        y propiedades del material.
        """
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))  # Crea una figura con 4 subplots
        
        # Gráfico 1: Distribución de elementos
        elementos_unicos, counts = np.unique(self.elementos, return_counts=True)
        axs[0, 0].bar(elementos_unicos, counts, color='skyblue', edgecolor='navy')
        axs[0, 0].set_title("Distribución de Elementos", fontweight='bold')
        axs[0, 0].set_xlabel("Elementos")
        axs[0, 0].set_ylabel("Frecuencia")
        axs[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Gráfico 2: Masa vs Carga
        scatter = axs[0, 1].scatter(self.masas, self.cargas, c=self.electronegatividades, cmap='viridis', s=50)
        axs[0, 1].set_title("Masa vs Carga", fontweight='bold')
        axs[0, 1].set_xlabel("Masa (u)")
        axs[0, 1].set_ylabel("Carga (e)")
        axs[0, 1].grid(linestyle='--', alpha=0.7)
        plt.colorbar(scatter, ax=axs[0, 1], label='Electronegatividad')
        
        # Gráfico 3: Distribución de electronegatividades
        axs[1, 0].hist(self.electronegatividades, bins=20, color='lightgreen', edgecolor='darkgreen')
        axs[1, 0].set_title("Distribución de Electronegatividades", fontweight='bold')
        axs[1, 0].set_xlabel("Electronegatividad")
        axs[1, 0].set_ylabel("Frecuencia")
        axs[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Gráfico 4: Propiedades del material
        propiedades = [self.modulo_young, self.conductividad_termica, self.densidad, self.temperatura_fusion]
        nombres = ['Módulo de Young\n(GPa)', 'Conductividad\nTérmica\n(W/(m·K))', 'Densidad\n(g/cm³)', 'Temperatura\nde Fusión\n(K)']
        x = range(len(nombres))
        bars = axs[1, 1].bar(x, propiedades, color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'])
        axs[1, 1].set_title("Propiedades del Material", fontweight='bold')
        axs[1, 1].set_xticks(x)
        axs[1, 1].set_xticklabels(nombres)
        axs[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Añade etiquetas de valores a las barras
        for bar in bars:
            height = bar.get_height()
            axs[1, 1].annotate(f'{height:.2f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),  # 3 points vertical offset
                               textcoords="offset points",
                               ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

    def visualizar_dinamica_molecular(self, temperaturas, energias):
        """
        Visualiza los resultados de la simulación de dinámica molecular.
        Muestra dos gráficos: evolución de la temperatura y evolución de la energía total.
        
        :param temperaturas: Lista de temperaturas durante la simulación
        :param energias: Lista de energías durante la simulación
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))  # Crea una figura con 2 subplots verticales
        
        # Gráfico de temperatura
        ax1.plot(temperaturas, color='red')
        ax1.set_title('Evolución de la Temperatura', fontweight='bold')
        ax1.set_xlabel('Paso de Simulación')
        ax1.set_ylabel('Temperatura (K)')
        ax1.grid(linestyle='--', alpha=0.7)
        
        # Gráfico de energía
        ax2.plot(energias, color='blue')
        ax2.set_title('Evolución de la Energía Total', fontweight='bold')
        ax2.set_xlabel('Paso de Simulación')
        ax2.set_ylabel('Energía (J)')
        ax2.grid(linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()

    def analisis_espectral(self):
        """
        Realiza y visualiza el análisis espectral del Hamiltoniano.
        Muestra dos gráficos: espectro de energía y distribución de los primeros eigenvectores.
        """
        # Calcula los eigenvalores y eigenvectores del Hamiltoniano
        eigenvalores, eigenvectores = np.linalg.eigh(self.H)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # Crea una figura con 2 subplots horizontales
        
        # Gráfico del espectro de energía
        ax1.plot(eigenvalores, 'o-', color='purple')
        ax1.set_title('Espectro de Energía', fontweight='bold')
        ax1.set_xlabel('Índice del Estado')
        ax1.set_ylabel('Energía (J)')
        ax1.grid(linestyle='--', alpha=0.7)
        
        # Gráfico de la distribución de los primeros eigenvectores
        num_estados = min(5, self.num_particulas)
        for i in range(num_estados):
            ax2.plot(eigenvectores[:, i], label=f'Estado {i+1}')
        ax2.set_title('Primeros Estados Propios', fontweight='bold')
        ax2.set_xlabel('Índice de Partícula')
        ax2.set_ylabel('Amplitud')
        ax2.legend()
        ax2.grid(linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()

# Código de ejemplo para usar el simulador
simulador = SimuladorQuimico(num_particulas=50, tipo_red='FCC')
temperaturas, energias = simulador.simular_dinamica_molecular()

# Llamadas a los métodos de visualización
simulador.representacion_3d()
simulador.representacion_molecular_3d()
simulador.representacion_macroscopica()
simulador.visualizar_dinamica_molecular(temperaturas, energias)
simulador.analisis_espectral()