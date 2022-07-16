import open3d as o3d
import numpy as np
import random
import torch.utils.data as data

def convert_to_point_cloud(mesh,number_of_point_per_meter=50):
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.1, 0.1, 0.7]) 
    area = mesh.get_surface_area()
    number_of_point_all = int(area * number_of_point_per_meter)
    pcd = mesh.sample_points_uniformly(number_of_points=number_of_point_all)
    return pcd 
    

def point_cloud_torus(number_of_point_per_meter = 50,torus_radius=1.0, tube_radius=0.5, radial_resolution=30, tubular_resolution=20):
    mesh = o3d.geometry.TriangleMesh.create_torus(torus_radius=torus_radius, tube_radius=tube_radius, radial_resolution=radial_resolution, tubular_resolution=tubular_resolution)
    pcd = convert_to_point_cloud(mesh,number_of_point_per_meter = number_of_point_per_meter)
    return pcd
# 0.5 < r < 2
# 0.1 < tur < 2



def point_cloud_cylinder_half(number_of_point_per_meter=50 ,radius=1.0, height=2.0, resolution=20, split=4, create_uv_map=False):

    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=resolution, split=split, create_uv_map=create_uv_map)
    pcd = convert_to_point_cloud(mesh,number_of_point_per_meter = number_of_point_per_meter)
    points = np.array(pcd.points)
    size_cut = np.random.random() * (height)
    cut = size_cut - height / 2 
    
    points_select  = points[:,2] > cut
    points = points[points_select]
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.array(pcd.colors)[points_select])
    pcd.normals = o3d.utility.Vector3dVector(np.array(pcd.normals)[points_select])

    return pcd
# 1 < r < 2
# 2 < h < 10



def point_cloud_cylinder(number_of_point_per_meter=50 ,radius=1.0, height=2.0, resolution=20, split=4, create_uv_map=False):

    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=resolution, split=split, create_uv_map=create_uv_map)
    pcd = convert_to_point_cloud(mesh,number_of_point_per_meter = number_of_point_per_meter)

    return pcd


# 0.5 < r < 2
# 0.5 < h < 10

def point_cloud_cylinder_triangle(number_of_point_per_meter=50 ,radius=1.0, height=2.0, resolution=3, split=4, create_uv_map=False):

    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=resolution, split=split, create_uv_map=create_uv_map)
    pcd = convert_to_point_cloud(mesh,number_of_point_per_meter = number_of_point_per_meter)

    return pcd
# 0.5 < r < 2
# 0.5 < h < 10


def point_cloud_cylinder_triangle_half(number_of_point_per_meter=50 ,radius=1.0, height=2.0, resolution=3, split=4, create_uv_map=False):

    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=resolution, split=split, create_uv_map=create_uv_map)
    pcd = convert_to_point_cloud(mesh,number_of_point_per_meter = number_of_point_per_meter)
    points = np.array(pcd.points)
    size_cut = np.random.random() * (height)
    cut = size_cut - height / 2 
    
    points_select  = points[:,2] > cut
    points = points[points_select]
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.array(pcd.colors)[points_select])
    pcd.normals = o3d.utility.Vector3dVector(np.array(pcd.normals)[points_select])

    return pcd
# 0.5 < r < 2
# 0.5 < h < 10


def point_cloud_cylinder_fiveangle(number_of_point_per_meter=50 ,radius=1.0, height=2.0, resolution=5, split=4, create_uv_map=False):

    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=resolution, split=split, create_uv_map=create_uv_map)
    pcd = convert_to_point_cloud(mesh,number_of_point_per_meter = number_of_point_per_meter)

    return pcd
# 0.5 < r < 2
# 0.5 < h < 10


def point_cloud_cylinder_fiveangle_half(number_of_point_per_meter=50 ,radius=1.0, height=2.0, resolution=5, split=4, create_uv_map=False):

    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=resolution, split=split, create_uv_map=create_uv_map)
    pcd = convert_to_point_cloud(mesh,number_of_point_per_meter = number_of_point_per_meter)
    points = np.array(pcd.points)
    size_cut = np.random.random() * (height)
    cut = size_cut - height / 2 
    
    points_select  = points[:,2] > cut
    points = points[points_select]
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.array(pcd.colors)[points_select])
    pcd.normals = o3d.utility.Vector3dVector(np.array(pcd.normals)[points_select])

    return pcd
# 0.5 < r < 2
# 0.5 < h < 10

def point_cloud_cylinder_sixangle(number_of_point_per_meter=50 ,radius=1.0, height=2.0, resolution=6, split=4, create_uv_map=False):

    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=resolution, split=split, create_uv_map=create_uv_map)
    pcd = convert_to_point_cloud(mesh,number_of_point_per_meter = number_of_point_per_meter)

    return pcd
# 1 < r < 2
# 0.5 < h < 10




def point_cloud_cylinder_sixangle_half(number_of_point_per_meter=50 ,radius=1.0, height=2.0, resolution=6, split=4, create_uv_map=False):

    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=resolution, split=split, create_uv_map=create_uv_map)
    pcd = convert_to_point_cloud(mesh,number_of_point_per_meter = number_of_point_per_meter)
    points = np.array(pcd.points)
    size_cut = np.random.random() * (height)
    cut = size_cut - height / 2 
    
    points_select  = points[:,2] > cut
    points = points[points_select]
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.array(pcd.colors)[points_select])
    pcd.normals = o3d.utility.Vector3dVector(np.array(pcd.normals)[points_select])

    return pcd
# 1 < r < 2
# 0.5 < h < 10

def point_cloud_box_half(number_of_point_per_meter=50 ,radius=1.0, height=2.0, resolution=4, split=4, create_uv_map=False):

    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=resolution, split=split, create_uv_map=create_uv_map)
    pcd = convert_to_point_cloud(mesh,number_of_point_per_meter = number_of_point_per_meter)
    points = np.array(pcd.points)
    size_cut = np.random.random() * (height)
    cut = size_cut - height / 2 
    
    points_select  = points[:,2] > cut
    points = points[points_select]
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.array(pcd.colors)[points_select])
    pcd.normals = o3d.utility.Vector3dVector(np.array(pcd.normals)[points_select])

    return pcd
# 1 < r < 2
# 0.5 < h < 10

def point_cloud_tetrahedron(number_of_point_per_meter=50 , radius=1.0, create_uv_map=False):

    mesh = o3d.geometry.TriangleMesh.create_tetrahedron(radius=radius , create_uv_map=create_uv_map)
    pcd = convert_to_point_cloud(mesh,number_of_point_per_meter = number_of_point_per_meter)

    return pcd


# 1 < r < 4



def point_cloud_octahedron(number_of_point_per_meter=50 , radius=1.0, create_uv_map=False):

    mesh = o3d.geometry.TriangleMesh.create_octahedron(radius=radius , create_uv_map=create_uv_map)
    pcd = convert_to_point_cloud(mesh,number_of_point_per_meter = number_of_point_per_meter)

    return pcd

# 0.7 < r < 3

def point_cloud_box(number_of_point_per_meter=50,width=1.0, height=1.0, depth=1.0, create_uv_map=False, map_texture_to_each_face=False):
    mesh = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth, create_uv_map=create_uv_map, map_texture_to_each_face=map_texture_to_each_face)
    pcd = convert_to_point_cloud(mesh,number_of_point_per_meter = number_of_point_per_meter)
    return pcd
# 0.5 < w < 2
# 0.5 < h < 2
# 0.5 < d < 2

def point_cloud_mobius_zero(number_of_point_per_meter=50,length_split=50, width_split=15, twists=0, raidus=1, flatness=1, width=1, scale=1):
    mesh = o3d.geometry.TriangleMesh.create_mobius(length_split=length_split, width_split=width_split, twists=twists, raidus=raidus, flatness=flatness, width=width, scale=scale)
    pcd = convert_to_point_cloud(mesh,number_of_point_per_meter = number_of_point_per_meter)
    return pcd

# 0.5 < r < 2
# 0.5 < w < 2

def point_cloud_mobius_one(number_of_point_per_meter=50,length_split=50, width_split=15, twists=1, raidus=1, flatness=1, width=1, scale=1):
    mesh = o3d.geometry.TriangleMesh.create_mobius(length_split=length_split, width_split=width_split, twists=twists, raidus=raidus, flatness=flatness, width=width, scale=scale)
    pcd = convert_to_point_cloud(mesh,number_of_point_per_meter = number_of_point_per_meter)
    return pcd

# 0.5 < r < 2
# 0.5 < w < 2

def point_cloud_mobius_two(number_of_point_per_meter=50,length_split=50, width_split=15, twists=2, raidus=1, flatness=1, width=1, scale=1):
    mesh = o3d.geometry.TriangleMesh.create_mobius(length_split=length_split, width_split=width_split, twists=twists, raidus=raidus, flatness=flatness, width=width, scale=scale)
    pcd = convert_to_point_cloud(mesh,number_of_point_per_meter = number_of_point_per_meter)
    return pcd

# 0.5 < r < 2
# 0.5 < w < 2
def point_cloud_mobius_three(number_of_point_per_meter=50,length_split=50, width_split=15, twists=3, raidus=1, flatness=1, width=1, scale=1):
    mesh = o3d.geometry.TriangleMesh.create_mobius(length_split=length_split, width_split=width_split, twists=twists, raidus=raidus, flatness=flatness, width=width, scale=scale)
    pcd = convert_to_point_cloud(mesh,number_of_point_per_meter = number_of_point_per_meter)
    return pcd

# 0.5 < r < 2
# 0.5 < w < 2


def point_cloud_mobius_four_to_seven(number_of_point_per_meter=50,length_split=50, width_split=15, twists=7, raidus=1, flatness=1, width=1, scale=1):
    mesh = o3d.geometry.TriangleMesh.create_mobius(length_split=length_split, width_split=width_split, twists=twists, raidus=raidus, flatness=flatness, width=width, scale=scale)
    pcd = convert_to_point_cloud(mesh,number_of_point_per_meter = number_of_point_per_meter)
    return pcd

# 0.5 < r < 2
# 0.5 < w < 2
# 4 <= t <= 7  

def point_cloud_sphere(number_of_point_per_meter=50,radius=1.0, resolution=20, create_uv_map=False):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution, create_uv_map=create_uv_map)
    pcd = convert_to_point_cloud(mesh,number_of_point_per_meter = number_of_point_per_meter)
    return pcd
# 0.5 < r < 2

def point_cloud_sphere_half(number_of_point_per_meter=50,radius=1.0, resolution=20, create_uv_map=False):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution, create_uv_map=create_uv_map)
    pcd = convert_to_point_cloud(mesh,number_of_point_per_meter = number_of_point_per_meter)
    points = np.array(pcd.points)
    size_cut = np.random.random() * (radius)
    cut = size_cut - radius / 2 
    
    points_select  = points[:,2] > cut
    points = points[points_select]
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.array(pcd.colors)[points_select])
    pcd.normals = o3d.utility.Vector3dVector(np.array(pcd.normals)[points_select])
    return pcd
# 0.5 < r < 2

# def point_cloud_icosahedron(number_of_point_per_meter=50 , radius=1.0, create_uv_map=False):

#     mesh = o3d.geometry.TriangleMesh.create_icosahedron(radius=radius , create_uv_map=create_uv_map)
#     pcd = convert_to_point_cloud(mesh,number_of_point_per_meter = number_of_point_per_meter)

#     return pcd


def point_cloud_cone(number_of_point_per_meter=50,radius=1.0, height=2.0, resolution=20, split=1, create_uv_map=False):
    mesh = o3d.geometry.TriangleMesh.create_cone(radius=radius, height=height, resolution=resolution, split=split, create_uv_map=create_uv_map)
    pcd = convert_to_point_cloud(mesh,number_of_point_per_meter = number_of_point_per_meter)
    return pcd
# 0.5 < r < 1.5
# 0.5 < h < 2


def point_cloud_cone_half(number_of_point_per_meter=50,radius=1.0, height=2.0, resolution=20, split=1, create_uv_map=False):
    mesh = o3d.geometry.TriangleMesh.create_cone(radius=radius, height=height, resolution=resolution, split=split, create_uv_map=create_uv_map)
    pcd = convert_to_point_cloud(mesh,number_of_point_per_meter = number_of_point_per_meter)
    points = np.array(pcd.points)
    size_cut = np.random.random() * (height)
    cut = size_cut / 4
    
    points_select  = points[:,2] > cut
    points = points[points_select]
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.array(pcd.colors)[points_select])
    pcd.normals = o3d.utility.Vector3dVector(np.array(pcd.normals)[points_select])
    return pcd

# 0.5 < r < 1.5
# 0.75 < h < 2


# def point_cloud_torus_half(number_of_point_per_meter = 50,torus_radius=1.0, tube_radius=0.5, radial_resolution=30, tubular_resolution=20):
#     mesh = o3d.geometry.TriangleMesh.create_torus(torus_radius=torus_radius, tube_radius=tube_radius, radial_resolution=radial_resolution, tubular_resolution=tubular_resolution)
#     pcd = convert_to_point_cloud(mesh,number_of_point_per_meter = number_of_point_per_meter)
#     points = np.array(pcd.points)
#     size_cut = np.random.random() * (tube_radius * 1.25)
#     cut = tube_radius  - size_cut 
    
#     points_select  = points[:,2] > cut
#     points = points[points_select]
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(np.array(pcd.colors)[points_select])
#     pcd.normals = o3d.utility.Vector3dVector(np.array(pcd.normals)[points_select])
#     return pcd
# 0.5 < r < 2
# 0.2 < tur < 2

sh1 = point_cloud_torus
sh2 = point_cloud_cylinder_half
sh3 = point_cloud_cylinder
sh4 = point_cloud_cylinder_triangle
sh5 = point_cloud_cylinder_triangle_half
sh6 = point_cloud_cylinder_fiveangle
sh7 = point_cloud_cylinder_fiveangle_half
sh8 = point_cloud_cylinder_sixangle
sh9 = point_cloud_cylinder_sixangle_half
sh10 = point_cloud_box_half
sh11 = point_cloud_tetrahedron
sh12 = point_cloud_octahedron
sh13 =point_cloud_box
sh14 =point_cloud_mobius_zero
sh15 =point_cloud_mobius_one
sh16 =point_cloud_mobius_two
sh17 =point_cloud_mobius_three
sh18 =point_cloud_mobius_four_to_seven
sh19 =point_cloud_sphere
sh20 =point_cloud_sphere_half
sh21 =point_cloud_cone
sh22 =point_cloud_cone_half
# sh23 = point_cloud_torus_half
    
# o3d.visualization.draw_geometries([point_cloud_torus(torus_radius = 2,tube_radius=0.1 ),point_cloud_sphere(radius=0.5)])





def return_random_shape(shapes = [1,2,3,4], point_per_meter = 50):
    list_of_shape = []
    
    for shape_func in shapes:
        if shape_func == 1:
            # 0.5 < r < 2
            # 0.1 < tur < 2
            r = random.uniform(0.5, 2)
            tur = random.uniform(0.1, r - 0.1)
            list_of_shape.append(point_cloud_torus(number_of_point_per_meter = point_per_meter,
                                                   torus_radius= r, 
                                                   tube_radius=  tur,
                                                   radial_resolution=30,tubular_resolution=20))
        elif shape_func == 2:
            list_of_shape.append(point_cloud_cylinder_half(number_of_point_per_meter=point_per_meter ,
                                                           radius= random.uniform(1, 2), 
                                                           height= random.uniform(2, 7),
                                                           resolution=20, 
                                                           split=4, create_uv_map=False))
        elif shape_func == 3:
            list_of_shape.append(point_cloud_cylinder(number_of_point_per_meter=point_per_meter ,
                                                      radius= random.uniform(0.5, 2), 
                                                      height= random.uniform(0.5, 10), 
                                                      resolution=20, 
                                                      split=4, create_uv_map=False))
        elif shape_func == 4:
            list_of_shape.append(point_cloud_cylinder_triangle(number_of_point_per_meter=point_per_meter ,
                                                               radius= random.uniform(0.5, 2), 
                                                               height= random.uniform(0.5, 7),
                                                               resolution=3, 
                                                               split=4, create_uv_map=False))
        elif shape_func == 5:
            list_of_shape.append(point_cloud_cylinder_triangle_half(number_of_point_per_meter=point_per_meter ,
                                                               radius= random.uniform(0.5, 2), 
                                                               height= random.uniform(0.5, 7),
                                                               resolution=3, 
                                                               split=4, create_uv_map=False))

        elif shape_func == 6:
            list_of_shape.append(point_cloud_cylinder_fiveangle(number_of_point_per_meter=point_per_meter ,
                                                               radius= random.uniform(0.5, 2), 
                                                               height= random.uniform(0.5, 7),
                                                               resolution=5, 
                                                               split=4, create_uv_map=False))
        elif shape_func == 7:
            list_of_shape.append(point_cloud_cylinder_fiveangle_half(number_of_point_per_meter=point_per_meter ,
                                                               radius= random.uniform(0.5, 2), 
                                                               height= random.uniform(0.5, 7),
                                                               resolution=5, 
                                                               split=4, create_uv_map=False))
        elif shape_func == 8:
            list_of_shape.append(point_cloud_cylinder_sixangle(number_of_point_per_meter=point_per_meter ,
                                                               radius= random.uniform(0.5, 2), 
                                                               height= random.uniform(0.5, 7),
                                                               resolution=6, 
                                                               split=4, create_uv_map=False))
        elif shape_func == 9:
            list_of_shape.append(point_cloud_cylinder_sixangle_half(number_of_point_per_meter=point_per_meter ,
                                                               radius= random.uniform(0.5, 2), 
                                                               height= random.uniform(0.5, 7),
                                                               resolution=6, 
                                                               split=4, create_uv_map=False))
        elif shape_func == 10:
            list_of_shape.append(point_cloud_box_half(number_of_point_per_meter=point_per_meter ,
                                                               radius= random.uniform(0.5, 2), 
                                                               height= random.uniform(0.5, 7),
                                                               resolution=4, 
                                                               split=4, create_uv_map=False))
            
        elif shape_func == 11:
            list_of_shape.append(point_cloud_tetrahedron(number_of_point_per_meter=point_per_meter , 
                                                        radius=random.uniform(1, 4),
                                                        create_uv_map=False))
            
        elif shape_func == 12:
            list_of_shape.append(point_cloud_octahedron(number_of_point_per_meter=point_per_meter , 
                                                        radius=random.uniform(0.7, 3),
                                                        create_uv_map=False))
        elif shape_func == 13:
            list_of_shape.append(point_cloud_box(number_of_point_per_meter=point_per_meter,
                                                 width=random.uniform(0.5, 2.5),
                                                 height=random.uniform(0.5, 4),
                                                 depth=random.uniform(0.5, 4),
                                                 create_uv_map=False, 
                                                 map_texture_to_each_face=False))

        elif shape_func == 14:
            list_of_shape.append(point_cloud_mobius_zero(number_of_point_per_meter=point_per_meter,
                                                                  length_split=50,
                                                                  width_split=15, 
                                                                  twists=0,
                                                                  raidus=random.uniform(0.5, 2), 
                                                                  flatness=1, 
                                                                  width=random.uniform(0.5, 2), 
                                                                  scale=1))
        elif shape_func == 15:
            list_of_shape.append(point_cloud_mobius_one(number_of_point_per_meter=point_per_meter,
                                                                  length_split=50,
                                                                  width_split=15, 
                                                                  twists=1,
                                                                  raidus=random.uniform(0.5, 2), 
                                                                  flatness=1, 
                                                                  width=random.uniform(0.5, 2), 
                                                                  scale=1))
        elif shape_func == 16:
            list_of_shape.append(point_cloud_mobius_two(number_of_point_per_meter=point_per_meter,
                                                                  length_split=50,
                                                                  width_split=15, 
                                                                  twists=2,
                                                                  raidus=random.uniform(0.5, 2), 
                                                                  flatness=1, 
                                                                  width=random.uniform(0.5, 2), 
                                                                  scale=1))
        elif shape_func == 17:
            list_of_shape.append(point_cloud_mobius_three(number_of_point_per_meter=point_per_meter,
                                                                  length_split=50,
                                                                  width_split=15, 
                                                                  twists=3,
                                                                  raidus=random.uniform(0.5, 2), 
                                                                  flatness=1, 
                                                                  width=random.uniform(0.5, 2), 
                                                                  scale=1))
        elif shape_func == 18:
            list_of_shape.append(point_cloud_mobius_four_to_seven(number_of_point_per_meter=point_per_meter,
                                                                  length_split=50,
                                                                  width_split=15, 
                                                                  twists=random.randint(4,7),
                                                                  raidus=random.uniform(0.5, 2), 
                                                                  flatness=1, 
                                                                  width=random.uniform(0.5, 2), 
                                                                  scale=1))
        elif shape_func == 19:
            list_of_shape.append(point_cloud_sphere(number_of_point_per_meter=point_per_meter ,
                                                               radius= random.uniform(0.5, 2), 
                                                               resolution=4, 
                                                               create_uv_map=False))
        elif shape_func == 20:
            list_of_shape.append(point_cloud_sphere_half(number_of_point_per_meter=point_per_meter,
                                                         radius=random.uniform(0.5, 2), 
                                                         resolution=20,
                                                         create_uv_map=False))
        elif shape_func == 21:
            
            list_of_shape.append(point_cloud_cone(number_of_point_per_meter=point_per_meter ,
                                                               radius= random.uniform(0.5, 1.5), 
                                                               height= random.uniform(0.5, 2),
                                                               resolution=4, 
                                                               split=4, create_uv_map=False))
        elif shape_func == 22:
            list_of_shape.append(point_cloud_cone_half(number_of_point_per_meter=point_per_meter,
                                                       radius=random.uniform(0.5, 1.5),
                                                       height=random.uniform(0.75, 2),
                                                       resolution=20, split=1, 
                                                       create_uv_map=False))

    return list_of_shape
            
            
            
# o3d.visualization.draw_geometries(return_random_shape([5,1,4,2,3,2,11]))
def random_point(pc,color_select = None):
    
    pc_points = np.array(pc.points)
    pc_points_size = len(pc_points)
    rand_n = np.random.randint(0,pc_points_size)
    rand_p = pc.points[rand_n]
    if color_select != None:
        colors = np.array(pc.colors)
        colors[rand_n] = color_select
    pc.colors = o3d.utility.Vector3dVector(colors)
    p_normal = np.array(pc.normals)[rand_n]
    
    return pc,rand_p,rand_n,p_normal

def concate_pc_to_base(base_pc,pc_list,base_number,shapes_side_numbers):


    
    
    all_distinct = []
    for pc in pc_list:
        base_pc , rand_p_source , rand_n_source, p_normal_source =random_point(base_pc,[1,0,0])
        pc , rand_p_attach , rand_n_attach, p_normal_attach =random_point(pc,[1,0,0])
        
        noise = (np.random.rand(1,3) /10)[0]
        angles = np.arccos(p_normal_source)+noise
        
        RO = pc.get_rotation_matrix_from_xyz((angles[0],angles[1],angles[2]))
        pc = pc.rotate(RO, center=(0,0,0))
        
        pc  = pc.translate((rand_p_source[0]-rand_p_attach[0], rand_p_source[1]-rand_p_attach[1] ,rand_p_source[2]-rand_p_attach[2]), relative=True)
        all_distinct.append(pc)
    points_base_size  = np.array(base_pc.points).shape[0]
    labels = np.zeros(points_base_size) + base_number

    for i,pc in zip(shapes_side_numbers , all_distinct):
        points_base  = np.array(base_pc.points)
        colors_base  = np.array(base_pc.colors)
        normals_base = np.array(base_pc.normals)
        points_another  = np.array(pc.points)
        colors_another   = np.array(pc.colors)
        color1 = np.random.random()
        color2 = np.random.random()
        color3 = np.random.random()
        colors_another = np.zeros(colors_another.shape) + [color1,color2,color3]
        labels_another = np.zeros(points_another.shape[0]) + i


        
        normals_another = np.array(pc.normals)
        
        points_base  = np.concatenate((points_base,points_another))
        colors_base  = np.concatenate((colors_base,colors_another))
        normals_base = np.concatenate((normals_base,normals_another))
        labels = np.concatenate((labels,labels_another))
        base_pc.points = o3d.utility.Vector3dVector(points_base)
        base_pc.colors = o3d.utility.Vector3dVector(colors_base)
        base_pc.normals = o3d.utility.Vector3dVector(normals_base)
        

        
    points_base  = np.array(base_pc.points)
    points_base =  points_base + np.random.rand(points_base.shape[0],points_base.shape[1])/10
    base_pc.points = o3d.utility.Vector3dVector(points_base)
#     o3d.visualization.draw_geometries([base_pc])

    max_bound = abs (base_pc.get_max_bound()).max()
    min_bound = abs (base_pc.get_min_bound()).max()
    max_for_scale = max(max_bound,min_bound)
    scale_p = 1/max_for_scale
    
    base_pc = base_pc.scale(scale_p,np.array([0,0,0]))
#     print(base_pc.get_min_bound())
#     print(base_pc.get_max_bound())

    # o3d.visualization.draw_geometries([base_pc])

    return base_pc,labels
    
    

       
def generate_point_cloud(number_of_point_down_sample = 3000):        
    base_number = random.randint(1,22)
    number_of_shaps = random.randint(1,7)
    shapes_side_numbers = []
    for _ in range(number_of_shaps):
        shapes_side_numbers.append(random.randint(1,22))
    base_shape = return_random_shape(shapes=[base_number],point_per_meter=500)[0] 
    sides      =    return_random_shape(shapes= shapes_side_numbers,point_per_meter=500)
    pc , labels= concate_pc_to_base(base_shape,sides,base_number,shapes_side_numbers)
    points = np.array(pc.points)
#     print(points.shape,labels.shape)
    idx = np.random.choice(np.arange(len(points)), number_of_point_down_sample, replace=False)
    points = points[idx]
    labels = labels[idx]
   

    return points,labels

# import timeit

# start = timeit.default_timer()

# #Your statements here

      
# for i in range(10):
#     points , labels = generate_point_cloud()
#     np.savetxt('obj'+str(i)+'.txt',points)
#     np.savetxt('labels'+str(i)+'.txt',labels)
        
# stop = timeit.default_timer()

# print('Time: ', stop - start)    
    
class ShapeNetPart(data.Dataset):
    NUM_SEGMENTATION_CLASSES = 22
    POINT_DIMENSION = 3



    def __init__(self,partition = 'trainval',number_of_data = 10000,num_points=2500,class_choice = []):

        self.number_of_points = num_points
        self.number_of_data = number_of_data
        self.seg_num_all = 22
        self.seg_start_index = 0


    def __getitem__(self, index):

        points , labels = generate_point_cloud()
        points = np.round(points, 8)
      

        return points,[random.randint(0,15)],labels



    def __len__(self):
        return self.number_of_data


    
     