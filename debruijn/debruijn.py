#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""Perform assembly based on debruijn graph."""

import argparse
import os
import sys
import numpy as np
from pathlib import Path
from networkx import (
    DiGraph,
    all_simple_paths,
    lowest_common_ancestor,
    has_path,
    random_layout,
    draw,
    spring_layout,
)
import matplotlib
from operator import itemgetter
import random

random.seed(9001)
from random import randint
import statistics
import textwrap
import matplotlib.pyplot as plt
from typing import Iterator, Dict, List

matplotlib.use("Agg")

__author__ = "Your Name"
__copyright__ = "Universite Paris Diderot"
__credits__ = ["Your Name"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Your Name"
__email__ = "your@email.fr"
__status__ = "Developpement"


def isfile(path: str) -> Path:  # pragma: no cover
    """Check if path is an existing file.

    :param path: (str) Path to the file

    :raises ArgumentTypeError: If file does not exist

    :return: (Path) Path object of the input file
    """
    myfile = Path(path)
    if not myfile.is_file():
        if myfile.is_dir():
            msg = f"{myfile.name} is a directory."
        else:
            msg = f"{myfile.name} does not exist."
        raise argparse.ArgumentTypeError(msg)
    return myfile


def get_arguments():  # pragma: no cover
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description=__doc__, usage="{0} -h".format(sys.argv[0])
    )
    parser.add_argument(
        "-i", dest="fastq_file", type=isfile, required=True, help="Fastq file"
    )
    parser.add_argument(
        "-k", dest="kmer_size", type=int, default=22, help="k-mer size (default 22)"
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=Path,
        default=Path(os.curdir + os.sep + "contigs.fasta"),
        help="Output contigs in fasta file (default contigs.fasta)",
    )
    parser.add_argument(
        "-f", dest="graphimg_file", type=Path, help="Save graph as an image (png)"
    )
    return parser.parse_args()


def read_fastq(fastq_file: Path) -> Iterator[str]:
    """Extract reads from fastq files.

    :param fastq_file: (Path) Path to the fastq file.
    :return: A generator object that iterate the read sequences.
    """
    with open(fastq_file, "r") as fq:
        for line in fq:
            if line.startswith("@"):
                yield next(fq).strip() # Yield the sequence


def cut_kmer(read: str, kmer_size: int) -> Iterator[str]:
    """Cut read into kmers of size kmer_size.

    :param read: (str) Sequence of a read.
    :return: A generator object that provides the kmers (str) of size kmer_size.
    """
    for i in range(len(read) - kmer_size + 1):
        yield read[i:i+kmer_size]

def build_kmer_dict(fastq_file: Path, kmer_size: int) -> Dict[str, int]:
    """Build a dictionnary object of all kmer occurrences in the fastq file

    :param fastq_file: (str) Path to the fastq file.
    :return: A dictionnary object that identify all kmer occurrences.
    """
    kmer_occurence = {}
    for read in read_fastq(fastq_file):
        for kmer in cut_kmer(read, kmer_size):
            if kmer in kmer_occurence:
                kmer_occurence[kmer] += 1
            else:
                kmer_occurence[kmer] = 1
    return kmer_occurence


def build_graph(kmer_dict: Dict[str, int]) -> DiGraph:
    """Build the debruijn graph

    :param kmer_dict: A dictionnary object that identify all kmer occurrences.
    :return: A directed graph (nx) of all kmer substring and weight (occurrence).
    """
    graph = DiGraph()
    # Get nodes and arcs from kmer
    for kmer, weight in kmer_dict.items():
        prefix = kmer[:-1]
        sufix = kmer[1:]
        graph.add_edge(prefix, sufix, weight = weight)
    return graph


def remove_paths(
    graph: DiGraph,
    path_list: List[List[str]],
    delete_entry_node: bool,
    delete_sink_node: bool,
) -> DiGraph:
    """Remove a list of path in a graph. A path is set of connected node in
    the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    for path in path_list:
        if not delete_entry_node:
            path = path[1:]
        if not delete_sink_node:
            path = path[:-1]
        graph.remove_nodes_from(path)
    return graph


def select_best_path(
    graph: DiGraph,
    path_list: List[List[str]],
    path_length: List[int],
    weight_avg_list: List[float],
    delete_entry_node: bool = False,
    delete_sink_node: bool = False,
) -> DiGraph:
    """Select the best path between different paths

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param path_length_list: (list) A list of length of each path
    :param weight_avg_list: (list) A list of average weight of each path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    sd = statistics.stdev(weight_avg_list)

    if sd > 0:
        ind_best = weight_avg_list.index(max(weight_avg_list))
    else:
        sd = statistics.stdev(path_length)
        if sd > 0:
            ind_best = path_length.index(max(path_length))
        else:
            ind_best = randint(0, len(path_list) - 1)

    del path_list[ind_best]

    modified_graph = remove_paths(graph, path_list, delete_entry_node, delete_sink_node)

    return modified_graph
    



def path_average_weight(graph: DiGraph, path: List[str]) -> float:
    """Compute the weight of a path

    :param graph: (nx.DiGraph) A directed graph object
    :param path: (list) A path consist of a list of nodes
    :return: (float) The average weight of a path
    """
    return statistics.mean(
        [d["weight"] for (u, v, d) in graph.subgraph(path).edges(data=True)]
    )


def solve_bubble(graph: DiGraph, ancestor_node: str, descendant_node: str) -> DiGraph:
    """Explore and solve bubble issue

    :param graph: (nx.DiGraph) A directed graph object
    :param ancestor_node: (str) An upstream node in the graph
    :param descendant_node: (str) A downstream node in the graph
    :return: (nx.DiGraph) A directed graph object
    """
    bubble_path = list(all_simple_paths(graph, ancestor_node, descendant_node))
    average_weight = [path_average_weight(graph, path) for path in bubble_path]
    length_list = [len(path) for path in bubble_path]

    graph = select_best_path(graph, bubble_path, length_list, average_weight, delete_entry_node=False, delete_sink_node=False)

    return graph


def simplify_bubbles(graph: DiGraph) -> DiGraph:
    """Detect and explode bubbles

    :param graph: (nx.DiGraph) A directed graph object
    :return: (nx.DiGraph) A directed graph object
    """
    bubble = False
    for node in graph.nodes:
        predecessors_list = list(graph.predecessors(node))
        if len(predecessors_list) > 1:
            for i in range(len(predecessors_list)):
                node_i = predecessors_list[i]
                for j in range(i+1, len(predecessors_list)):
                    node_j = predecessors_list[j]
                    ancestor = lowest_common_ancestor(graph, node_i, node_j)
                    if ancestor != None:
                        bubble = True
                        break
        if bubble == True:
            break
    if bubble:
        graph = simplify_bubbles(solve_bubble(graph, ancestor,node))

    return graph



def solve_entry_tips(graph: DiGraph, starting_nodes: List[str]) -> DiGraph:
    """Remove entry tips

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of starting nodes
    :return: (nx.DiGraph) A directed graph object
    """
    for node in graph.nodes:
        predecessors_list = list(graph.predecessors(node))
        if len(predecessors_list) > 1:
            entry_path = []
            lengths = []
            weights = []
            for entry in starting_nodes:
                if has_path(graph, entry, node):
                    entry_path += (all_simple_paths(graph, entry, node))
                
            for path in entry_path:
                lengths.append(len(path))
                weights.append(path_average_weight(graph, path))

            if len(entry_path) > 1:
                graph = select_best_path(graph, entry_path, lengths, weights, delete_entry_node = True, delete_sink_node = False)
                return solve_entry_tips(graph, get_starting_nodes(graph))
        
    return graph



def solve_out_tips(graph: DiGraph, ending_nodes: List[str]) -> DiGraph:
    """Remove out tips

    :param graph: (nx.DiGraph) A directed graph object
    :param ending_nodes: (list) A list of ending nodes
    :return: (nx.DiGraph) A directed graph object
    """
    for node in graph.nodes:
        successors_list = list(graph.successors(node))
        if len(successors_list) > 1:
            out_path = []
            lengths = []
            weights = []
            for out in ending_nodes:
                if has_path(graph, node, out):
                    out_path += (all_simple_paths(graph, node, out))
                
            for path in out_path:
                lengths.append(len(path))
                weights.append(path_average_weight(graph, path))

            if len(out_path) > 1:
                graph = select_best_path(graph, out_path, lengths, weights, delete_entry_node = False, delete_sink_node = True)
                return solve_out_tips(graph, get_sink_nodes(graph))
        
    return graph



def get_starting_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without predecessors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without predecessors
    """
    starting_nodes = []
    for node in graph.nodes:
        if not list(graph.predecessors(node)):
            starting_nodes.append(node)
    return starting_nodes



def get_sink_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without successors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without successors
    """
    sink_nodes = []
    for node in graph.nodes:
        if not list(graph.successors(node)):
            sink_nodes.append(node)
    return sink_nodes


def get_contigs(
    graph: DiGraph, starting_nodes: List[str], ending_nodes: List[str]
) -> List:
    """Extract the contigs from the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of nodes without predecessors
    :param ending_nodes: (list) A list of nodes without successors
    :return: (list) List of [contiguous sequence and their length]
    """
    contig_list = []
    for start in starting_nodes:
        for sink in ending_nodes:
            if has_path(graph, start, sink):
                for path in all_simple_paths(graph, start, sink):
                    contig = path[0]
                    for node in path[1:]:
                        contig += node[-1]
                    contig_list.append((contig, len(contig)))
    return contig_list


def save_contigs(contigs_list: List[str], output_file: Path) -> None:
    """Write all contigs in fasta format

    :param contig_list: (list) List of [contiguous sequence and their length]
    :param output_file: (Path) Path to the output file
    """
    with open(output_file,"w+") as fasta:
        for i in range(len(contigs_list)):
            contig = contigs_list[i]
            sequence = contig[0]
            length = contig[1]
            fasta.write(f">contig_{i} len={length}\n")
            fasta.write(f"{sequence}\n") 


def draw_graph(graph: DiGraph, graphimg_file: Path) -> None:  # pragma: no cover
    """Draw the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param graphimg_file: (Path) Path to the output file
    """
    fig, ax = plt.subplots()
    elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] > 3]
    # print(elarge)
    esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] <= 3]
    # print(elarge)
    # Draw the graph with networkx
    # pos=nx.spring_layout(graph)
    pos = nx.random_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=6)
    nx.draw_networkx_edges(graph, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(
        graph, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )
    # nx.draw_networkx(graph, pos, node_size=10, with_labels=False)
    # save image
    plt.savefig(graphimg_file.resolve())


# ==============================================================
# Main program
# ==============================================================
def main() -> None:  # pragma: no cover
    """
    Main program function
    """
    # Get arguments
    args = get_arguments()

    FILE = args.fastq_file
    KMER_SIZE = args.kmer_size
    OUTPUT = args.output_file

    GRAPH = build_graph(build_kmer_dict(FILE, KMER_SIZE))

    GRAPH = simplify_bubbles(GRAPH)

    GRAPH = solve_entry_tips(GRAPH, get_starting_nodes(GRAPH))

    GRAPH = solve_out_tips(GRAPH, get_sink_nodes(GRAPH))

    save_contigs(get_contigs(GRAPH, get_starting_nodes(GRAPH), get_sink_nodes(GRAPH)), output_file = OUTPUT)




if __name__ == "__main__":  # pragma: no cover
    main()
