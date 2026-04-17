
from dataclasses import dataclass, field

from mminf.graph.base import DynamicLoop, GraphNode, GraphSection, Parallel, Sequential


@dataclass
class IterIndexTree:
    label: str
    iter_index: int = 0
    children: dict[str, "IterIndexTree"] = field(default_factory=dict)
    descendent_labels: set[str] = field(default_factory=set)

    def _set_desc_labels(self):
        for child in self.children.values():
            self.descendent_labels.add(child.label)
            self.descendent_labels.update(child.descendent_labels)

    def label_context_gt(self, other: "IterIndexTree", target_label: str) -> bool:
        """
        Whether the iter indices of "self" are greater than the indices of
        "other", specifically in the path leading up to (but not including)
        "target_label".

        For instance, if we are stopping the loop "target_label" but don't
        want to double-stop it, we can keep track of the last time it was
        stopped and only stop it again `new_time.label_context_gt(prev, label)`.
        """
        if self.label == target_label:
            return False
        assert self.label == other.label and target_label in self.descendent_labels
        if self.iter_index != other.iter_index:
            return self.iter_index > other.iter_index

        # iter indices are equal, so have to recurse down the path
        child_name = None
        child = None
        for label, child in self.children.items():
            if label in other.children and target_label in child.descendent_labels:
                child_name = label
                break
        if child_name is None:
            return False
        return child.label_context_gt(other.children[child_name], target_label)


def build_loop_index_tree(
    graph: GraphSection, fwd_idx: str
) -> IterIndexTree:
    root = IterIndexTree(label="_fwd_idx", iter_index=fwd_idx)

    def _build(graph: GraphSection) -> dict[str, IterIndexTree]:
        if isinstance(graph, GraphNode) or graph is None:
            return {}
        if isinstance(graph, Sequential) or isinstance(graph, Parallel):
            res = {}
            for sec in graph.sections:
                res.update(_build(sec))
            return res

        # otherwise, this is a loop
        label = graph.name if isinstance(graph, DynamicLoop) else graph._uuid_label
        base_node = IterIndexTree(
            label=label, iter_index=graph.curr_iter
        )
        base_node.children = _build(graph._curr_iter_section)
        base_node._set_desc_labels()
        return {label: base_node}
    root.children = _build(graph)
    root._set_desc_labels()
    return root

def update_loop_index_tree(
    index_tree: IterIndexTree, graph: GraphSection, fwd_idx: str
):
    index_tree.iter_index = fwd_idx

    def _update(tree: IterIndexTree, graph: GraphSection) -> dict[str, IterIndexTree]:
        if isinstance(graph, GraphNode) or graph is None:
            return
        if isinstance(graph, Sequential) or isinstance(graph, Parallel):
            for sec in graph.sections:
                _update(tree, sec)
                return

        # otherwise, this is a loop
        label = graph.name if isinstance(graph, DynamicLoop) else graph._uuid_label
        if label not in tree.children:
            return
        tree = tree.children[label]
        tree.iter_index = graph.curr_iter
        _update(tree, graph._curr_iter_section)
    _update(index_tree, graph)
