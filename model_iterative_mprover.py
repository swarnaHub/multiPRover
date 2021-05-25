from pytorch_transformers import BertPreTrainedModel, RobertaConfig, \
    ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP, RobertaModel
from pytorch_transformers.modeling_roberta import RobertaClassificationHead
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment


class RobertaForRR(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForRR, self).__init__(config)

        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, position_ids=None,
                head_mask=None):
        outputs = self.roberta(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                               attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            qa_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (qa_loss,) + outputs

        return outputs  # qa_loss, logits, (hidden_states), (attentions)


class NodeClassificationHead(nn.Module):

    def __init__(self, config):
        super(NodeClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class EdgeClassificationHead(nn.Module):

    def __init__(self, config):
        super(EdgeClassificationHead, self).__init__()
        self.dense = nn.Linear(3 * config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ProofEncodingLayer(nn.Module):

    def __init__(self, config):
        super(ProofEncodingLayer, self).__init__()
        self.encoder_layer_node = nn.TransformerEncoderLayer(d_model=config.hidden_size,
                                                             nhead=config.num_attention_heads)
        self.encoder_node = nn.TransformerEncoder(self.encoder_layer_node, num_layers=1)

        self.encoder_layer_edge = nn.TransformerEncoderLayer(d_model=3 * config.hidden_size,
                                                             nhead=config.num_attention_heads)
        self.encoder_edge = nn.TransformerEncoder(self.encoder_layer_edge, num_layers=1)

        self.param1 = nn.parameter.Parameter(torch.nn.init.xavier_uniform_(torch.empty(26, 26**2)))
        self.param2 = nn.parameter.Parameter(torch.nn.init.xavier_uniform_(torch.empty(3 * config.hidden_size, config.hidden_size)))
        self.linear = nn.Linear(2 * config.hidden_size, config.hidden_size)

        self.classifier_node = NodeClassificationHead(config)
        self.classifier_edge = EdgeClassificationHead(config)

    def forward(self, node_embeddings, edge_embeddings, node_label, edge_label, num_labels_node, num_labels_edge,
                num_proofs, loss_map, index, is_end):
        next_node_embeddings = self.encoder_node(node_embeddings)

        edge_conditioned_node_embeddings = torch.bmm(torch.bmm(self.param1.repeat(node_embeddings.shape[0], 1, 1), edge_embeddings), self.param2.repeat(node_embeddings.shape[0], 1, 1))
        next_node_embeddings = self.linear(torch.cat((next_node_embeddings, edge_conditioned_node_embeddings), dim=2))

        node_logits = self.classifier_node(next_node_embeddings)

        next_edge_embeddings = self.encoder_edge(edge_embeddings)

        edge_logits = self.classifier_edge(next_edge_embeddings)

        loss_fct = CrossEntropyLoss()

        # If the end of proof sequence is reached, no need to compute loss with all gold proofs
        if is_end:
            node_loss = loss_fct(node_logits.view(-1, num_labels_node), node_label[:, index, :].reshape(-1))
            edge_loss = loss_fct(edge_logits.view(-1, num_labels_edge), edge_label[:, index, :].reshape(-1))
            proof_loss = node_loss + edge_loss
            return loss_map, proof_loss, node_logits, edge_logits, next_node_embeddings, next_edge_embeddings
        else:
            # Calculate loss with all gold proofs
            for i in range(num_proofs):
                node_loss = loss_fct(node_logits.view(-1, num_labels_node), node_label[:, i, :].reshape(-1))
                edge_loss = loss_fct(edge_logits.view(-1, num_labels_edge), edge_label[:, i, :].reshape(-1))

                proof_loss = node_loss + edge_loss

                loss_map[(index, i)] = proof_loss
            # Proof loss ignored because this goes as part of the Hungarian computation
            return loss_map, None, node_logits, edge_logits, next_node_embeddings, next_edge_embeddings


class RobertaForRRIterativeMprover(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, max_proof):
        super(RobertaForRRIterativeMprover, self).__init__(config)

        self.num_labels = config.num_labels
        self.num_labels_node = 2
        self.num_labels_edge = 2
        self.num_proofs = max_proof
        self.roberta = RobertaModel(config)
        self.naf_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = RobertaClassificationHead(config)
        self.classifier_node = NodeClassificationHead(config)
        self.classifier_edge = EdgeClassificationHead(config)
        self.proof_layers = nn.ModuleList([ProofEncodingLayer(config) for i in range(self.num_proofs - 1)])

        self.apply(self.init_weights)

    def _get_hungarian_loss(self, loss_map, proof_count):
        cost_matrix = np.zeros((proof_count, proof_count))
        for i in range(proof_count):
            for j in range(proof_count):
                cost_matrix[i][j] = loss_map[(i, j)]

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        hungarian_loss = None
        for (pred_id, gold_id) in zip(row_ind, col_ind):
            if hungarian_loss is None:
                hungarian_loss = loss_map[(pred_id, gold_id)]
            else:
                hungarian_loss += loss_map[(pred_id, gold_id)]

        return hungarian_loss

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, proof_offset=None, node_label=None,
                edge_label=None, labels=None, proof_count=None, position_ids=None, head_mask=None):
        outputs = self.roberta(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                               attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]
        cls_output = sequence_output[:, 0, :]
        naf_output = self.naf_layer(cls_output)
        logits = self.classifier(sequence_output)

        max_node_length = node_label.shape[2]
        max_edge_length = edge_label.shape[2]
        batch_size = node_label.shape[0]
        embedding_dim = sequence_output.shape[2]

        batch_node_embedding = torch.zeros((batch_size, max_node_length, embedding_dim)).to("cuda")
        batch_edge_embedding = torch.zeros((batch_size, max_edge_length, 3 * embedding_dim)).to("cuda")

        for batch_index in range(batch_size):
            prev_index = 1
            sample_node_embedding = None
            count = 0
            for offset in proof_offset[batch_index]:
                if offset == 0:
                    break
                else:
                    rf_embedding = torch.mean(sequence_output[batch_index, prev_index:(offset + 1), :],
                                              dim=0).unsqueeze(0)
                    prev_index = offset + 1
                    count += 1
                    if sample_node_embedding is None:
                        sample_node_embedding = rf_embedding
                    else:
                        sample_node_embedding = torch.cat((sample_node_embedding, rf_embedding), dim=0)

            # Add the NAF output at the end
            sample_node_embedding = torch.cat((sample_node_embedding, naf_output[batch_index].unsqueeze(0)), dim=0)

            repeat1 = sample_node_embedding.unsqueeze(0).repeat(len(sample_node_embedding), 1, 1)
            repeat2 = sample_node_embedding.unsqueeze(1).repeat(1, len(sample_node_embedding), 1)
            sample_edge_embedding = torch.cat((repeat1, repeat2, (repeat1 - repeat2)), dim=2)

            sample_edge_embedding = sample_edge_embedding.view(-1, sample_edge_embedding.shape[-1])

            # Append 0s at the end (these will be ignored for loss)
            sample_node_embedding = torch.cat((sample_node_embedding,
                                               torch.zeros((max_node_length - count - 1, embedding_dim)).to("cuda")),
                                              dim=0)
            sample_edge_embedding = torch.cat((sample_edge_embedding,
                                               torch.zeros((max_edge_length - len(sample_edge_embedding),
                                                            3 * embedding_dim)).to("cuda")), dim=0)

            batch_node_embedding[batch_index, :, :] = sample_node_embedding
            batch_edge_embedding[batch_index, :, :] = sample_edge_embedding

        node_logits = self.classifier_node(batch_node_embedding)
        edge_logits = self.classifier_edge(batch_edge_embedding)

        all_node_logits, all_edge_logits = None, None

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            qa_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            all_proof_loss = None

            loss_map = {}
            for batch_index in range(batch_size):
                sample_node_logits = node_logits[batch_index, :, :].unsqueeze(0)
                sample_edge_logits = edge_logits[batch_index, :, :].unsqueeze(0)

                # First proof layer
                # Take loss before "end of graph sequence"
                # The proofs before "end of graph sequence" can appear in any order
                for i in range(proof_count[batch_index]):
                    node_loss = loss_fct(node_logits[batch_index, :, :].view(-1, self.num_labels_node),
                                         node_label[batch_index, i, :].reshape(-1))
                    edge_loss = loss_fct(edge_logits[batch_index, :, :].view(-1, self.num_labels_edge),
                                         edge_label[batch_index, i, :].reshape(-1))

                    # Index notation is (pred_id, gold_id)
                    loss_map[(0, i)] = node_loss + edge_loss

                # Next proof layers
                temp_node_embedding = batch_node_embedding[batch_index, :, :].unsqueeze(0)
                temp_edge_embedding = batch_edge_embedding[batch_index, :, :].unsqueeze(0)
                for i in range(1, self.num_proofs):
                    if i < proof_count[batch_index]:
                        is_end = False
                    else:
                        is_end = True
                    loss_map, proof_loss, curr_node_logits, curr_edge_logits, next_node_embeddings, next_edge_embeddings \
                        = self.proof_layers[i - 1](temp_node_embedding, temp_edge_embedding,
                                                   node_label[batch_index, :, :].unsqueeze(0),
                                                   edge_label[batch_index, :, :].unsqueeze(0),
                                                   self.num_labels_node, self.num_labels_edge, proof_count[batch_index],
                                                   loss_map, i, is_end)

                    temp_node_embedding, temp_edge_embedding = next_node_embeddings, next_edge_embeddings
                    sample_node_logits = torch.cat((sample_node_logits, curr_node_logits), dim=0)
                    sample_edge_logits = torch.cat((sample_edge_logits, curr_edge_logits), dim=0)

                    if is_end:
                        if all_proof_loss is None:
                            all_proof_loss = proof_loss
                        else:
                            all_proof_loss += proof_loss

                hungarian_loss = self._get_hungarian_loss(loss_map, proof_count[batch_index])
                all_proof_loss += hungarian_loss

                if all_node_logits is None:
                    all_node_logits = sample_node_logits.unsqueeze(0)
                    all_edge_logits = sample_edge_logits.unsqueeze(0)
                else:
                    all_node_logits = torch.cat((all_node_logits, sample_node_logits.unsqueeze(0)), dim=0)
                    all_edge_logits = torch.cat((all_edge_logits, sample_edge_logits.unsqueeze(0)), dim=0)

            total_loss = qa_loss + all_proof_loss
            outputs = (total_loss, qa_loss, all_proof_loss, logits, all_node_logits, all_edge_logits) + outputs

        return outputs
