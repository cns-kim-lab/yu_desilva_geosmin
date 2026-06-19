import os 
import flybrains
from get_mesh_neuron import *
from fafbseg import flywire
from meshparty import trimesh_vtk
import vtk
from multiprocessing import Pool
from tqdm import tqdm
import pickle
import numpy as np 
from color_utils import hex_to_rgb 




### for creating scale bar - cube look like a square form on the projection the view 
def create_scalebar_at_wanted_pos(pos,spec):
    import vtk
    cube_source = vtk.vtkCubeSource()
    cube_source.SetCenter(pos[0],pos[1],pos[2])  # Set the center of the cube
    cube_source.SetXLength(spec[0])     # Set the length of the cube along the x-axis
    cube_source.SetYLength(spec[1])     # Set the length of the cube along the y-axis
    cube_source.SetZLength(spec[2])     # Set the length of the cube along the z-axis
    
    # Create a mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(cube_source.GetOutputPort())
    
    # Create an actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    # Create a property to set the  color of the cube source
    cube_color = vtk.vtkProperty()
    cube_color.SetColor(0,0,0)  
    
    # Set the property to the actor
    actor.SetProperty(cube_color)
    return actor

def make_mesh_actor(mesh,color_rgb,opacity=1):
    mesh_actor_each_cell = trimesh_vtk.mesh_actor(mesh,color=color_rgb,opacity=opacity)
    return mesh_actor_each_cell 


def make_projection_image(save_path,actors_list,center,scalebar=False,projection_view='xy',do_save=True):

    if projection_view == 'xy':
        backoff_vector=[0,0,1]
        up_vector = [0,-1,0]
    elif projection_view == 'yz':
        backoff_vector=[-1,0,0]
        up_vector = [0,-1,0]
    elif projection_view == 'xz':
        backoff_vector = [0,-1,0]
        up_vector = [0,0,-1]

    camera = trimesh_vtk.oriented_camera(center=center ,      # focus point
        backoff=700,
        backoff_vector=backoff_vector,
        up_vector = up_vector)  
    camera.ParallelProjectionOn()
    camera.SetParallelScale(70000)
    if do_save:
        trimesh_vtk.render_actors(actors_list,camera=camera,do_save=do_save,filename=f'{save_path}.png')
    else:
        trimesh_vtk.render_actors(actors_list,camera=camera,do_save=do_save)