# -*- coding: utf-8 -*-
"""Caja de herramientas de ArcGIS Pro para las salidas del módulo `inundacion`.

Carga en el mapa activo el tirante horario (NetCDF con tiempo), el tirante máximo y
los inmuebles por nivel, y deja la capa lista para el Time Slider. No hace
geoprocesamiento hidrológico: el modelo corre en Python
(`python -m inundacion simular ...`) y aquí solo se visualiza.

Para el video: pestaña Animación > Importar > Time Slider Steps > Exportar > Video.
"""
import os

import arcpy

COLORES_NIVEL = {"Alto": [192, 57, 43, 100], "Medio": [230, 126, 34, 100], "Sin afectación": [138, 147, 158, 100]}


class Toolbox(object):
    def __init__(self):
        self.label = "Inundación por huracán"
        self.alias = "inundacion"
        self.tools = [CargarEscenario]


class CargarEscenario(object):
    def __init__(self):
        self.label = "Cargar escenario de inundación"
        self.description = ("Agrega al mapa el tirante horario (con tiempo), el tirante máximo y los inmuebles "
                            "de un escenario generado con `python -m inundacion simular`.")
        self.canRunInBackground = False

    def getParameterInfo(self):
        carpeta = arcpy.Parameter(displayName="Carpeta de salidas del escenario", name="carpeta",
                                  datatype="DEFolder", parameterType="Required", direction="Input")
        escenario = arcpy.Parameter(displayName="Escenario (prefijo de archivos)", name="escenario",
                                    datatype="GPString", parameterType="Required", direction="Input")
        escenario.value = "escenario0"
        return [carpeta, escenario]

    def execute(self, parameters, messages):
        carpeta = parameters[0].valueAsText
        escenario = parameters[1].valueAsText
        mapa = arcpy.mp.ArcGISProject("CURRENT").activeMap
        if mapa is None:
            raise arcpy.ExecuteError("Abre una vista de mapa antes de correr la herramienta.")

        # addDataFromPath no abre el NetCDF ("Failed to add data"); como ráster multidimensional sí,
        # y toma la dimensión time para el Time Slider.
        nombre = escenario + " · tirante horario (m)"
        resultado = arcpy.md.MakeMultidimensionalRasterLayer(
            os.path.join(carpeta, escenario + "_tirante_horario.nc"), nombre, variables=["tirante"])
        capa_tiempo = mapa.addLayer(resultado.getOutput(0))[0]
        _rampa_azules(capa_tiempo)

        capa_max = mapa.addDataFromPath(os.path.join(carpeta, escenario + "_tirante_max.tif"))
        capa_max.name = escenario + " · tirante máximo (m)"
        capa_max.visible = False
        _rampa_azules(capa_max)

        capa_inm = mapa.addDataFromPath(os.path.join(carpeta, escenario + "_inmuebles.gpkg", "main.inmuebles"))
        capa_inm.name = escenario + " · inmuebles"
        _simbolos_nivel(capa_inm)

        if capa_tiempo.supports("TIME") and capa_tiempo.isTimeEnabled:
            messages.addMessage("Capa con tiempo lista para el Time Slider.")
        else:
            messages.addWarningMessage("La capa no quedó con tiempo; habilítalo en Propiedades > Tiempo.")


def _rampa_azules(capa):
    """La simbología es cosmética: si falla, la capa se queda con la de omisión."""
    try:
        sim = capa.symbology
        sim.updateColorizer("RasterStretchColorizer")
        rampas = arcpy.mp.ArcGISProject("CURRENT").listColorRamps("Blues*")
        if rampas:
            sim.colorizer.colorRamp = rampas[0]
        capa.symbology = sim
    except Exception as e:
        arcpy.AddWarning("No pude aplicar la rampa de azules: {}".format(e))


def _simbolos_nivel(capa):
    try:
        sim = capa.symbology
        sim.updateRenderer("UniqueValueRenderer")
        sim.renderer.fields = ["nivel"]
        for grupo in sim.renderer.groups:
            for item in grupo.items:
                valor = item.values[0][0]
                if valor in COLORES_NIVEL:
                    item.symbol.color = {"RGB": COLORES_NIVEL[valor]}
                    item.symbol.size = 4 if valor == "Sin afectación" else 7
        capa.symbology = sim
    except Exception as e:
        arcpy.AddWarning("No pude simbolizar los inmuebles por nivel: {}".format(e))
