import numpy as np
import argparse
import json
import os
import sys
from scipy.optimize import linear_sum_assignment

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proof_utils import get_proof_graph, get_proof_graph_with_fail


def get_node_edge_label(proofs, sentence_scramble, nfact, nrule):
    component_index_map = {}
    for (i, index) in enumerate(sentence_scramble):
        if index <= nfact:
            component = "triple" + str(index)
        else:
            component = "rule" + str(index - nfact)
        component_index_map[component] = i
    component_index_map["NAF"] = nfact + nrule

    gold_proof_count = len(proofs.split("OR"))

    all_node_label = np.zeros((gold_proof_count, nfact + nrule + 1), dtype=int)
    all_edge_label = np.zeros((gold_proof_count, nfact + nrule + 1, nfact + nrule + 1), dtype=int)

    for (k, proof) in enumerate(proofs.split("OR")):
        # print(proof)

        if "FAIL" in proof:
            nodes, edges = get_proof_graph_with_fail(proof)
        else:
            nodes, edges = get_proof_graph(proof)
        # print(nodes)
        # print(edges)

        for node in nodes:
            index = component_index_map[node]
            all_node_label[k][index] = 1

        edges = list(set(edges))
        for edge in edges:
            start_index = component_index_map[edge[0]]
            end_index = component_index_map[edge[1]]
            all_edge_label[k][start_index][end_index] = 1

    return all_node_label, all_edge_label


def get_gold_labels_and_proofs(data_dir):
    test_file = os.path.join(data_dir, "dev.jsonl")
    meta_test_file = os.path.join(data_dir, "meta-dev.jsonl")

    f1 = open(test_file, "r", encoding="utf-8-sig")
    f2 = open(meta_test_file, "r", encoding="utf-8-sig")

    gold_nodes, gold_edges = [], []
    gold_labels = []
    cnt = 0
    for record, meta_record in zip(f1, f2):
        print(cnt)
        cnt += 1
        record = json.loads(record)
        meta_record = json.loads(meta_record)
        # if not record["id"].startswith("AttPosBirdsVar2"):
        #    continue

        sentence_scramble = record["meta"]["sentenceScramble"]
        for (j, question) in enumerate(record["questions"]):
            meta_data = meta_record["questions"]["Q" + str(j + 1)]

            proofs = meta_data["proofs"]
            nfact = meta_record["NFact"]
            nrule = meta_record["NRule"]
            label = question["label"]
            # if question["meta"]["QDep"] != 3:
            #    continue
            all_node_indices, all_edge_indices = get_node_edge_label(proofs, sentence_scramble, nfact, nrule)
            gold_nodes.append(all_node_indices)
            gold_edges.append(all_edge_indices)
            gold_labels.append(label)

    return gold_nodes, gold_edges, gold_labels

def get_precision_recall_f1(gold_nodes=None, pred_nodes=None, gold_edges=None, pred_edges=None):
    gold_count = len(gold_nodes) if gold_nodes is not None else len(gold_edges)
    pred_count = len(pred_nodes) if pred_nodes is not None else len(pred_edges)

    if gold_nodes is not None and gold_edges is not None:
        cost_matrix = np.zeros((len(gold_nodes), len(pred_nodes)))
        for i in range(len(gold_nodes)):
            for j in range(len(pred_nodes)):
                if (gold_nodes[i] != pred_nodes[j]).sum() > 0 or (gold_edges[i] != pred_edges[j]).sum():
                    cost_matrix[i][j] = 1

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        correct_count = 0
        for (gold_id, pred_id) in zip(row_ind, col_ind):
            if np.array_equal(gold_nodes[gold_id], pred_nodes[pred_id]) and np.array_equal(gold_edges[gold_id], pred_edges[pred_id]):
                correct_count += 1
    elif gold_nodes is not None:
        cost_matrix = np.zeros((len(gold_nodes), len(pred_nodes)))
        for i in range(len(gold_nodes)):
            for j in range(len(pred_nodes)):
                if (gold_nodes[i] != pred_nodes[j]).sum() > 0:
                    cost_matrix[i][j] = 1

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        correct_count = 0
        for (gold_id, pred_id) in zip(row_ind, col_ind):
            if np.array_equal(gold_nodes[gold_id], pred_nodes[pred_id]):
                correct_count += 1
    else:
        cost_matrix = np.zeros((len(gold_edges), len(pred_edges)))
        for i in range(len(gold_edges)):
            for j in range(len(pred_edges)):
                if (gold_edges[i] != pred_edges[j]).sum() > 0:
                    cost_matrix[i][j] = 1

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        correct_count = 0
        for (gold_id, pred_id) in zip(row_ind, col_ind):
            if np.array_equal(gold_edges[gold_id], pred_edges[pred_id]):
                correct_count += 1

    print(correct_count)
    print(gold_count)
    print(pred_count)
    print("\n")

    precision = correct_count / pred_count
    recall = correct_count / gold_count
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return precision, recall, f1


def get_score(all_gold_nodes, all_pred_nodes, all_gold_edges, all_pred_edges, all_pred_labels, all_gold_labels):
    print(len(all_gold_nodes))
    print(len(all_pred_nodes))
    print(len(all_pred_edges))
    assert len(all_gold_nodes) == len(all_pred_nodes)
    assert len(all_gold_edges) == len(all_pred_edges)
    assert len(all_gold_nodes) == len(all_gold_edges)
    assert len(all_pred_nodes) == len(all_pred_edges)
    print(len(all_gold_labels))
    print(len(all_pred_labels))
    assert len(all_pred_labels) == len(all_gold_labels)

    correct_qa = 0
    overall_node_precision, overall_node_recall, overall_node_f1 = 0., 0., 0.
    overall_edge_precision, overall_edge_recall, overall_edge_f1 = 0., 0., 0.
    overall_proof_precision, overall_proof_recall, overall_proof_f1 = 0., 0., 0.
    full_correct = 0
    for cnt, (gold_nodes, pred_nodes, gold_edges, pred_edges, gold_label, pred_label) in enumerate(zip(all_gold_nodes, all_pred_nodes, all_gold_edges, all_pred_edges, all_gold_labels, all_pred_labels)):
        print(cnt)

        is_correct_qa = False
        if str(gold_label) == pred_label:
            is_correct_qa = True
            correct_qa += 1

        node_precision, node_recall, node_f1 = get_precision_recall_f1(gold_nodes=gold_nodes, pred_nodes=pred_nodes)
        edge_precision, edge_recall, edge_f1 = get_precision_recall_f1(gold_edges=gold_edges, pred_edges=pred_edges)
        proof_precision, proof_recall, proof_f1 = get_precision_recall_f1(gold_nodes=gold_nodes, pred_nodes=pred_nodes,
                                                                          gold_edges=gold_edges, pred_edges=pred_edges)


        overall_node_precision += node_precision
        overall_node_recall += node_recall
        overall_node_f1 += node_f1

        overall_edge_precision += edge_precision
        overall_edge_recall += edge_recall
        overall_edge_f1 += edge_f1

        overall_proof_precision += proof_precision
        overall_proof_recall += proof_recall
        overall_proof_f1 += proof_f1

        if proof_f1 == 1.0 and is_correct_qa:
            full_correct += 1

    return {
        "QA_accuracy": correct_qa / len(all_gold_nodes),
        "node_precision": overall_node_precision / len(all_gold_nodes),
        "node_recall": overall_node_recall / len(all_gold_nodes),
        "node_f1": overall_node_f1 / len(all_gold_nodes),
        "edge_precision": overall_edge_precision / len(all_gold_nodes),
        "edge_recall": overall_edge_recall / len(all_gold_nodes),
        "edge_f1": overall_edge_f1 / len(all_gold_nodes),
        "proof_precision": overall_proof_precision / len(all_gold_nodes),
        "proof_recall": overall_proof_recall / len(all_gold_nodes),
        "proof_f1": overall_proof_f1 / len(all_gold_nodes),
        "full_accuracy": full_correct / len(all_gold_nodes)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--qa_pred_file", default=None, type=str, required=True)
    parser.add_argument("--node_pred_file", default=None, type=str, required=True)
    parser.add_argument("--edge_pred_file", default=None, type=str, required=True)

    args = parser.parse_args()
    with open(args.qa_pred_file, "r", encoding="utf-8-sig") as f:
        all_pred_labels = f.read().splitlines()

    all_gold_nodes, all_gold_edges, all_gold_labels = get_gold_labels_and_proofs(args.data_dir)

    all_pred_nodes = []
    count = 0
    with open(args.node_pred_file, "r", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
        sample_pred_nodes = []
        for (k, line) in enumerate(lines):
            if line == "":
                sample_pred_nodes = np.array(sample_pred_nodes)
                all_pred_nodes.append(sample_pred_nodes)
                sample_pred_nodes = []
            else:
                pred_nodes = [int(x) for x in line[1:-1].split(",")]
                sample_pred_nodes.append(pred_nodes)

    all_pred_edges = []
    with open(args.edge_pred_file, "r", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
        sample_pred_edges = []
        sample_id = 0
        for (k, line) in enumerate(lines):
            if line == "":
                node_count = all_gold_edges[sample_id].shape[1]
                binarized_sample_pred_edges = np.zeros((len(sample_pred_edges), node_count, node_count), dtype=int)
                for (t, pred_edges) in enumerate(sample_pred_edges):
                    for pred_edge in pred_edges:
                        binarized_sample_pred_edges[t][pred_edge[0]][pred_edge[1]] = 1

                all_pred_edges.append(binarized_sample_pred_edges)
                sample_pred_edges = []
                sample_id += 1
            else:
                pred_edges = []
                if line != "[]":
                    edges = line[2:-2].split("), (")
                    for edge in edges:
                        edge = edge.split(", ")
                        pred_edges.append((int(edge[0]), int(edge[1])))

                sample_pred_edges.append(pred_edges)

    score = get_score(all_gold_nodes, all_pred_nodes, all_gold_edges, all_pred_edges, all_pred_labels, all_gold_labels)
    print(score)
