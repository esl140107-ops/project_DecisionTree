#include "visualization.h"
#include <stdio.h>

static int _node_id = 0;

//рекурсивная функция для генерации описания узлов и связей
static void _dot_recursive(const TreeNode *node, FILE *f, int id, TaskType task) {
    if (!node) return;
    
    if (node->is_leaf) {
        if (task == TASK_CLASSIFICATION) {
            fprintf(f, "  n%d [label=\"Класс: %d\\nОбъектов: %d\", shape=box, style=filled, fillcolor=\"#a1d99b\"];\n", 
                    id, (int)node->predicted_value, node->num_samples);
        } else {
            fprintf(f, "  n%d [label=\"Значение: %.3f\\nОбъектов: %d\", shape=box, style=filled, fillcolor=\"#fdbb84\"];\n", 
                    id, node->predicted_value, node->num_samples);
        }
    } else {
        fprintf(f, "  n%d [label=\"Признак X[%d] <= %.3f\\nImpurity: %.3f\", style=filled, fillcolor=\"#bcbddc\"];\n", 
                id, node->feature_index, node->threshold, node->impurity);
        
        int left_id = ++_node_id;
        int right_id = ++_node_id;
        
        _dot_recursive(node->left, f, left_id, task);
        _dot_recursive(node->right, f, right_id, task);
        
        fprintf(f, "  n%d -> n%d [label=\"Да\", color=\"#3182bd\", penwidth=1.5];\n", id, left_id);
        fprintf(f, "  n%d -> n%d [label=\"Нет\", color=\"#e6550d\", penwidth=1.5];\n", id, right_id);
    }
}
//функция для экспорта дерева 
void export_tree_to_dot(const TreeModel *model, const char *filename) {
    if (!model || !model->root) return;
    FILE *f = fopen(filename, "w");
    if (!f) return;
    
    fprintf(f, "digraph DecisionTree {\n");
    fprintf(f, "  node [fontname=\"Arial\", fontsize=10];\n");
    fprintf(f, "  edge [fontname=\"Arial\", fontsize=9];\n");
    
    _node_id = 0;
    _dot_recursive(model->root, f, 0, model->task);
    
    fprintf(f, "}\n");
    fclose(f);
    printf("[System] Граф дерева сохранен в %s\n", filename);
}

static void _console_recursive(const TreeNode *node, int depth, TaskType task) {
    if (!node) return;
    for (int i = 0; i < depth; i++) printf("  | ");
    
    if (node->is_leaf) {
        if (task == TASK_CLASSIFICATION) printf("=> Класс: %d (Объектов: %d)\n", (int)node->predicted_value, node->num_samples);
        else printf("=> Значение: %.3f (Объектов: %d)\n", node->predicted_value, node->num_samples);
    } else {
        printf("[X%d <= %.3f]\n", node->feature_index, node->threshold);
        _console_recursive(node->left, depth + 1, task);
        _console_recursive(node->right, depth + 1, task);
    }
}

void print_tree_console(const TreeModel *model) {
    if (!model) return;
    printf("\n=== Структура дерева ===\n");
    _console_recursive(model->root, 0, model->task);
    printf("========================\n");
}
