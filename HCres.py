from qiskit_metal.qlibrary.core import QComponent
from qiskit_metal import Dict, draw
import numpy as np
import sys
import numpy.typing as npt
import gmsh
import matplotlib.pyplot as plt

class HCresonator(QComponent):
    """
    Inherits QComponent Class

    Generates an 8 sided polygon resonator with N (working on making it take N as an input, for now itll be hard coded) capacitive fingers. 

    It is to note that thenumber of fingers within the capacitive cavity of the resonator would follow this relationship: 
    (p.cap_section_height)/(p.finger_width + p.finger_gap) = p.fingers_n
    I.e. to have a set amount of fingers one can use the relationship above to adjust any one parameter. 

    """

    """Component MetaData"""
    component_metadata = Dict(
        short_name ="HCres", _qgeometry_table_poly = "True", _qgeometry_table_path = "True"
    )

    # Below are the default options, which can be changed
    default_options = Dict(
        res_width = "20um", 
        finger_length = "17um",
        fingers_n = '7',
        finger_width = '2um', 
        finger_gap = '1um',
        geo_ind_length = '5um', 
        geo_ind_thick = '0.25um',
        metal_width = '5um', 
        res_height = '30um',
        cap_section_height = '21um', 
        layer = '1', 
        gap = '1um',
        trace_width = '0.25um',
        trace_gap = '1um',
    )

    def make(self): 
        # Method that builds the component
        p = self.p 
        N = int(p.fingers_n)

        # Fail-safe statements for Geometry: 
        if p.res_width <= p.finger_length: 
            raise ValueError(f"Your capacitor's finger length {p.finger_length} is larger than the resonator's width {p.res_width}")
        if N*(p.finger_width + p.finger_gap) > p.cap_section_height: 
            raise ValueError(f"You either have your capacitor's too wide and/or there's too much of them: {N*(p.finger_width + p.finger_gap)} vs {p.cap_section_height}")
        if p.res_height <= p.cap_section_height: 
            raise ValueError(f"Your Capacitive section height {p.cap_section_height} is larger than the height of the resonator {p.res_height}")
        if N*(p.finger_width + p.finger_gap) <= p.cap_section_height/2: 
            raise ValueError(f'You have far too few fingers to be reasonable {N*(p.finger_width + p.finger_gap)} vs {p.cap_section_height/2}')

        # Making shell of the resonator 
        body = draw.Polygon([
            (-p.geo_ind_length/2, 0),
            (+p.geo_ind_length/2, 0), 
            ( p.res_width/2 + p.metal_width, -(p.res_height - p.cap_section_height)/2),
            (p.res_width/2 + p.metal_width, -(p.res_height - p.cap_section_height)/2 - p.cap_section_height),
            (+p.geo_ind_length/2, -p.res_height),
            (-p.geo_ind_length/2, -p.res_height), 
            (- p.res_width/2 - p.metal_width, -(p.res_height - p.cap_section_height)/2 - p.cap_section_height),
            (- p.res_width/2 - p.metal_width, -(p.res_height - p.cap_section_height)/2),
            (-p.geo_ind_length/2, 0)
        ])

        # Etch out middle of our shell for capacitors 
        etch = draw.Polygon([
            (-p.geo_ind_length/2, -p.geo_ind_thick),
            (+p.geo_ind_length/2, -p.geo_ind_thick), 
            (+ p.res_width/2, -(p.res_height - p.cap_section_height)/2),
            ( + p.res_width/2, -(p.res_height - p.cap_section_height)/2 - p.cap_section_height),
            (+p.geo_ind_length/2, -p.res_height + p.geo_ind_thick),
            (-p.geo_ind_length/2, -p.res_height + p.geo_ind_thick), 
            (- p.res_width/2, -(p.res_height - p.cap_section_height)/2 - p.cap_section_height),
            (- p.res_width/2, -(p.res_height - p.cap_section_height)/2),
            (-p.geo_ind_length/2,-p.geo_ind_thick)
        ])

        body = draw.subtract(poly_main=body, poly_tool=etch)

        # Etching out the ground plane outside of our resonator 

        outer_etch = draw.Polygon([
            (-p.geo_ind_length/2, +p.gap - p.geo_ind_thick),
            (+p.geo_ind_length/2, +p.gap - p.geo_ind_thick), 
            (+ p.res_width/2 + p.metal_width + p.gap/np.sqrt(2), -(p.res_height - p.cap_section_height)/2 + p.gap/np.sqrt(2)),
            (+ p.res_width/2 + p.metal_width + p.gap/np.sqrt(2), -(p.res_height - p.cap_section_height)/2 - p.cap_section_height - p.gap/np.sqrt(2)),
            (+p.geo_ind_length/2, -p.res_height + p.geo_ind_thick - p.gap),
            (-p.geo_ind_length/2, -p.res_height + p.geo_ind_thick - p.gap), 
            (- p.res_width/2 - p.metal_width - p.gap/np.sqrt(2), -(p.res_height - p.cap_section_height)/2 - p.cap_section_height - p.gap/np.sqrt(2)) ,
            (- p.res_width/2 - p.metal_width - p.gap/np.sqrt(2), -(p.res_height - p.cap_section_height)/2 + p.gap/np.sqrt(2)),
            (-p.geo_ind_length/2, +p.gap - p.geo_ind_thick)
        ])

        outer_etch = draw.subtract(poly_main=outer_etch, poly_tool=body)

        # Making the Capacitive Fingers
        cap_fingers = []
        for i in range(N): 
            if i % 2 == 0: 
                finger = draw.rectangle(p.finger_length, p.finger_width, + p.res_width/2 - p.finger_length/2,
                                         -(p.res_height - p.cap_section_height)/2 - p.finger_width/2 - i * (p.finger_width + p.finger_gap))
                cap_fingers.append(finger)
            else: 
                finger = draw.rectangle(p.finger_length, p.finger_width, - p.res_width/2 + p.finger_length/2,
                                         -(p.res_height - p.cap_section_height)/2 - p.finger_width/2 - i * (p.finger_width + p.finger_gap))
                cap_fingers.append(finger)


        # Adding to qgeometry tables
        self.add_qgeometry("poly", {"body": body}, layer = p.layer)
        for i, finger in enumerate(cap_fingers):
            self.add_qgeometry("poly", {f"finger_{i}": finger}, layer=p.layer)
        self.add_qgeometry("poly", {"etch": etch}, layer=p.layer, subtract=True)
        self.add_qgeometry("poly", {"outer_etch": outer_etch}, layer=p.layer, subtract=True)

        self.add_pin('top_port', np.array([[0, -p.geo_ind_thick], [0, 0]]), width = p.geo_ind_thick, input_as_norm = False)

    def compute_flux(r0, m_dir = np.array([0,0,1.0]), dim = 2, tag = 2, nref = np.array([0,0,1.0])):
        node_tags, coords= gmsh.model.mesh.getNodesForPhysicalGroup(dim,tag)
        coords = np.asarray(coords).reshape(-1,3) * 1e-3
        idx = {t: i for i,t in enumerate(node_tags)}
        tris = []
        for ent in gmsh.model.getEntitiesForPhysicalGroup(dim,tag): 
            etypes, _, enodes = gmsh.model.mesh.getElements(dim,ent)
            for et, nn in zip(etypes, enodes): 
                if et == 2: 
                    tris.append(np.asarray(nn).reshape(-1,3))

        tris = np.vstack(tris)

        P = np.array([[idx[n] for n in row] for row in tris])

        p1, p2, p3 = coords[P[:,0]], coords[P[:,1]], coords[P[:,2]]
        centers = (p1 + p2 + p3)/ 3
        area = 0.5 * np.cross(p2 - p1, p3 - p1)
        area[area @ nref < 0] *= -1 
        mub = 9.274e-24 
        mu0 = (4* np.pi * 1e-7)
        g = 16.6 # I dont know its actual value

        m = (g * mub / 2) * (m_dir / np.linalg.norm(m_dir))

        d = centers - np.asarray(r0, dtype = float)
        rmag = np.linalg.norm(d, axis =1, keepdims=True)

        rhat = d / rmag

        B = (mu0 / (4*np.pi)) * (3 * rhat * (rhat @ m)[:,None] - m ) / rmag**3

        return np.einsum('ij, ij->i', B, area).sum()
    
    def get_area_of_res(self, dim: int = 2):
        gmsh.plugin.setNumber("MeshVolume", "Dimension", 2)
        gmsh.plugin.setNumber("MeshVolume", "PhysicalGroup", dim)
        gmsh.plugin.run("MeshVolume")

        views = gmsh.view.getTags()
        _, _, data = gmsh.view.getListData(views[-1])

        area = data[-1][-1]

        print(f"Metal Area of Resonator: {area} mm^2 = {area*1e6} um^2")

        return area 
    
    def return_verticies(self): 
        # This returns the verticies of the polygon at the centerline of our resonator and takes the Biot_sauvart line integra
        p = self.p
        geo_thickness = p.geo_ind_thick / 2
        geo_length = p.geo_ind_length / 2
        vert1 = [-geo_length, -geo_thickness]
        vert2 = [geo_length, -geo_thickness]

        capactive_length = p.res_width / 2 + p.metal_width / 2
        capactive_height = (p.res_height - p.cap_section_height)/2
        vert3 = [capactive_length, -capactive_height]
        vert4 = [capactive_length, -capactive_height - p.cap_section_height]

        vert5 = [geo_length, -p.res_height + geo_thickness]
        vert6 = [- geo_length, -p.res_height + geo_thickness ]        

        vert7 = [-capactive_length, -capactive_height - p.cap_section_height]
        vert8 = [-capactive_length, -capactive_height]

        verts = np.array([vert1, vert2, vert3, vert4, vert5, vert6, vert7, vert8])

        verts = np.vstack([verts, verts[0]])

        return verts

    def biot_savart(self, r0, I=1.0, N=1000):
        verts = self.return_verticies() * 1e-3
        verts = np.flip(verts, axis = 0)  # Flipping to have positive magnetic field in the resonator
        mu0 = 4e-7 * np.pi

        B = np.zeros(3)
        for i in range(len(verts) - 1):
            p1, p2 = verts[i], verts[i + 1]
            dl = p2 - p1
            t = np.linspace(0, 1, N)
            pts = p1[None, :] + t[:, None] * dl[None, :]

            R = np.column_stack([
                r0[0] - pts[:, 0],
                r0[1] - pts[:, 1],
                r0[2] * np.ones(N)
            ])

            dl3 = np.array([dl[0], dl[1], 0.0])
            cross = np.cross(dl3, R)
            integrand = cross / np.linalg.norm(R, axis=1)[:, None] ** 3
            B += np.trapz(integrand, t, axis=0)

        return mu0 * I / (4 * np.pi) * B
    
    def draw_verticies(self, ax=None, **plot_kwargs):
        verts = self.return_verticies()

        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')

        style = dict(color='red', marker='o', ms=4, lw=1.5, zorder=20)
        style.update(plot_kwargs)
        ax.plot(verts[:, 0], verts[:, 1], **style)

        for i, (x, y) in enumerate(verts[:-1]):
            ax.annotate(str(i + 1), (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=8, color='red')
        return ax