import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib as mpl
from ..core.em_functions import biot_savart_loop, compute_emf

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

class CatheterTrackingVisualizer:
    def __init__(self):
        # Initialize figure and 3D axis
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set up the layout to accommodate sliders and text
        plt.subplots_adjust(bottom=0.3, left=0.25)
        
        # Simulation parameters
        self.coil_radius = 0.05  # meters
        self.num_segments = 50
        self.current = 1.0  # Amperes
        self.frequency = 10000  # Hz
        self.num_turns = 100
        self.area = np.pi * (0.01)**2  # Cross-sectional area of Rx coil (m^2)
        
        # Tx coils configuration (3 coils evenly spaced along the ground, all facing upwards)
        self.tx_coils = [
            {'center': np.array([-0.1, -0.05, -0.08]), 'normal': np.array([0, 0, 1]), 'color': 'r'},
            {'center': np.array([0, 0.1, -0.08]), 'normal': np.array([0, 0, 1]), 'color': 'g'},
            {'center': np.array([0.1, -0.05, -0.08]), 'normal': np.array([0, 0, 1]), 'color': 'b'}
        ]
        
        # Rx coils configuration (2 coils sharing an axis)
        self.rx_center = np.array([0.1, 0.1, 0.1])  # Initial position
        self.rx_axis = np.array([1, 0, 0])  # Initial axis direction (will be normalized)
        self.rx_axis = self.rx_axis / np.linalg.norm(self.rx_axis)
        
        # Create sliders for Rx coil position and orientation (5 DOF)
        self.ax_x = plt.axes([0.25, 0.21, 0.65, 0.03])
        self.ax_y = plt.axes([0.25, 0.16, 0.65, 0.03])
        self.ax_z = plt.axes([0.25, 0.11, 0.65, 0.03])
        self.ax_theta = plt.axes([0.25, 0.06, 0.65, 0.03])
        self.ax_phi = plt.axes([0.25, 0.01, 0.65, 0.03])
        
        self.s_x = Slider(self.ax_x, 'X', -0.2, 0.2, valinit=self.rx_center[0])
        self.s_y = Slider(self.ax_y, 'Y', -0.2, 0.2, valinit=self.rx_center[1])
        self.s_z = Slider(self.ax_z, 'Z', -0.2, 0.2, valinit=self.rx_center[2])
        self.s_theta = Slider(self.ax_theta, 'θ', 0, 2*np.pi, valinit=0)
        self.s_phi = Slider(self.ax_phi, 'φ', 0, np.pi, valinit=np.pi/2)
        
        # Add reset button
        reset_ax = plt.axes([0.05, 0.25, 0.1, 0.04])
        self.reset_button = Button(reset_ax, 'Reset')
        
        # Text box for displaying EMF values
        self.emf_text = self.fig.text(0.02, 0.82, '', fontsize=10, 
                                    bbox=dict(facecolor='white', alpha=0.7))
        
        # Initialize visualization
        self.update_plot()
        
        # Connect sliders and button to update functions
        self.s_x.on_changed(self.update_sliders)
        self.s_y.on_changed(self.update_sliders)
        self.s_z.on_changed(self.update_sliders)
        self.s_theta.on_changed(self.update_sliders)
        self.s_phi.on_changed(self.update_sliders)
        self.reset_button.on_clicked(self.reset)
    
    def update_sliders(self, val):
        # Update Rx coil position and orientation based on sliders
        self.rx_center = np.array([self.s_x.val, self.s_y.val, self.s_z.val])
        
        # Update Rx coil axis based on spherical coordinates
        theta = self.s_theta.val
        phi = self.s_phi.val
        self.rx_axis = np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ])
        self.rx_axis = self.rx_axis / np.linalg.norm(self.rx_axis)
        
        # Update the plot
        self.update_plot()
    
    def reset(self, event):
        # Reset sliders to initial values
        self.s_x.reset()
        self.s_y.reset()
        self.s_z.reset()
        self.s_theta.reset()
        self.s_phi.reset()
    
    def draw_coil(self, center, normal, color, radius=None, num_points=30):
        if radius is None:
            radius = self.coil_radius
            
        # Generate points in a circle in the xy-plane
        theta = np.linspace(0, 2*np.pi, num_points)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = np.zeros_like(x)
        points = np.vstack((x, y, z)).T
        
        # Rotate points to align with the normal vector
        if not np.allclose(normal, [0, 0, 1]):
            v = np.cross([0, 0, 1], normal)
            c = np.dot([0, 0, 1], normal)
            k = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            R = np.eye(3) + k + k @ k * (1 / (1 + c + 1e-10))
            points = (R @ points.T).T
        
        # Translate to the center position
        points = points + center
        
        # Close the loop
        points = np.vstack((points, points[0]))
        
        return points
    
    def update_plot(self):
        self.ax.clear()
        
        # Draw Tx coils
        for i, coil in enumerate(self.tx_coils):
            points = self.draw_coil(coil['center'], coil['normal'], coil['color'])
            self.ax.plot(points[:, 0], points[:, 1], points[:, 2], 
                        color=coil['color'], linewidth=2, 
                        label=f'Tx {i+1}')
            
            # Draw normal vector
            arrow = Arrow3D(
                [coil['center'][0], coil['center'][0] + 0.1 * coil['normal'][0]],
                [coil['center'][1], coil['center'][1] + 0.1 * coil['normal'][1]],
                [coil['center'][2], coil['center'][2] + 0.1 * coil['normal'][2]],
                mutation_scale=15, lw=2, arrowstyle="-|>", color=coil['color']
            )
            self.ax.add_artist(arrow)
        
        # Draw Rx coils (2 coils sharing an axis)
        rx1_center = self.rx_center - 0.03 * self.rx_axis
        rx2_center = self.rx_center + 0.03 * self.rx_axis
        
        # Draw first Rx coil
        rx1_points = self.draw_coil(rx1_center, self.rx_axis, 'm', radius=0.01)
        self.ax.plot(rx1_points[:, 0], rx1_points[:, 1], rx1_points[:, 2], 
                    color='m', linewidth=2, label='Rx 1')
        
        # Draw second Rx coil
        rx2_points = self.draw_coil(rx2_center, self.rx_axis, 'c', radius=0.01)
        self.ax.plot(rx2_points[:, 0], rx2_points[:, 1], rx2_points[:, 2], 
                    color='c', linewidth=2, label='Rx 2')
        
        # Draw line connecting Rx coils
        self.ax.plot([rx1_center[0], rx2_center[0]],
                    [rx1_center[1], rx2_center[1]],
                    [rx1_center[2], rx2_center[2]], 'k--', alpha=0.5)
        
        # Calculate and display EMF for each Rx coil
        emf_text = []
        for i, rx_center in enumerate([rx1_center, rx2_center]):
            # Calculate total magnetic field at Rx coil from all Tx coils
            emf_text = []
            emf_matrix = np.zeros((2, 3))
            for i, rx_center in enumerate([rx1_center, rx2_center]):
                for j, tx in enumerate(self.tx_coils):
                    B = biot_savart_loop(
                        rx_center, 
                        tx['center'], 
                        self.coil_radius, 
                        tx['normal'], 
                        self.current, 
                        self.num_segments
                    )
                    emf = compute_emf(
                        B, 
                        self.rx_axis, 
                        self.area, 
                        self.num_turns, 
                        self.frequency
                    )
                    emf_matrix[i, j] = emf
                    emf_text.append(f'Rx {i+1} - Tx {j+1}: {emf:.2e} V')
        
        emf_vector = emf_matrix.flatten()
        
        # Update EMF text
        self.emf_text.set_text('\n'.join(emf_text))
        
        # Set plot properties
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('RF Catheter Tracking Simulation')
        self.ax.legend(loc='upper right')
        
        # Set equal aspect ratio
        self.ax.set_box_aspect([1, 1, 1])
        
        # Set axis limits
        self.ax.set_xlim([-0.2, 0.2])
        self.ax.set_ylim([-0.2, 0.2])
        self.ax.set_zlim([-0.2, 0.2])
        
        # Redraw
        self.fig.canvas.draw_idle()

def run_visualization():
    visualizer = CatheterTrackingVisualizer()
    plt.show()

if __name__ == "__main__":
    run_visualization()