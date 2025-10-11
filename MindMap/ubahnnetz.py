import networkx as nx
import matplotlib.pyplot as plt


class UBahnNetwork:
    def __init__(self):
        self.graph = nx.Graph()
        self.positions = {}  # Store fixed positions for each station
        self.line_colors = {}  # Store colors for each edge
        self._initialize_network()

    def _initialize_network(self):
        """Define U-Bahn stations and positions based on the BVG layout (simplified)."""
        stations = {
            "Krumme Lanke": (1, 5), "Onkel Toms Hütte": (2, 5), "Oskar-Helene-Heim": (3, 5),
            "Pankow": (8, 9), "Vinetastraße": (8, 8), "Schönhauser Allee": (8, 7),
            "Alexanderplatz": (6, 6), "Schillingstraße": (6, 5.5), "Strausberger Platz": (6, 5),
        }

        lines = {
            'U1': [('Krumme Lanke', 'Onkel Toms Hütte'), ('Onkel Toms Hütte', 'Oskar-Helene-Heim')],
            'U2': [('Pankow', 'Vinetastraße'), ('Vinetastraße', 'Schönhauser Allee')],
            'U5': [('Alexanderplatz', 'Schillingstraße'), ('Schillingstraße', 'Strausberger Platz')]
        }

        # Colors for each line (BVG-style)
        line_colors = {'U1': 'green', 'U2': 'red', 'U5': 'yellow'}

        for line, connections in lines.items():
            for station1, station2 in connections:
                self.graph.add_edge(station1, station2, line=line)
                self.positions[station1] = stations[station1]
                self.positions[station2] = stations[station2]
                self.line_colors[(station1, station2)] = line_colors[line]

    def rename_station(self, old_name, new_name):
        """Rename a station while keeping its connections."""
        if old_name in self.graph:
            self.graph = nx.relabel_nodes(self.graph, {old_name: new_name})
            self.positions[new_name] = self.positions.pop(old_name)
            print(f"Station '{old_name}' renamed to '{new_name}'")
        else:
            print(f"Station '{old_name}' not found in the network.")

    def show_network(self):
        """Display the U-Bahn network in BVG layout."""
        plt.figure(figsize=(8, 6))

        # Draw edges with their respective line colors
        for (station1, station2), color in self.line_colors.items():
            nx.draw_networkx_edges(self.graph, self.positions, edgelist=[(station1, station2)],
                                   width=4, edge_color=color, alpha=0.8)

        # Draw stations (nodes)
        nx.draw_networkx_nodes(self.graph, self.positions, node_color='white', edgecolors='black',
                               node_size=800, linewidths=1.5)

        # Draw labels
        nx.draw_networkx_labels(self.graph, self.positions, font_size=10, font_weight="bold")

        #plt.title("Berlin U-Bahn Network (BVG Layout)")
        plt.axis("off")
        plt.show()


# Example usage
if __name__ == "__main__":
    u_bahn = UBahnNetwork()
    u_bahn.show_network()  # Display the graph

    # Rename a station and re-display
    u_bahn.rename_station('Pankow', 'New Pankow')
    u_bahn.show_network()