#include "decision_tree.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define EPSILON 1e-9

/* 1. СТРУКТУРЫ УТИЛИТ И СОРТИРОВКА   */
typedef struct {
    int *indices;
    int size;
} Subset;

/**
 * @struct SortItem
 * @brief Хранит пару "Значение признака - Исходный индекс" для Quicksort.
 */
typedef struct {
    double value;
    int index;
} SortItem;

/**
 * @brief Компаратор для сортировки признаков по возрастанию.
 */
static int compare_items(const void *a, const void *b) {
    double v1 = ((SortItem*)a)->value;
    double v2 = ((SortItem*)b)->value;
    if (v1 < v2) return -1;
    if (v1 > v2) return 1;
    return 0;
}

static double calc_gini_from_counts(const int *counts, int num_classes, int total) {
    if (total == 0) return 0.0;
    double gini = 1.0;
    for (int i = 0; i < num_classes; i++) {
        if (counts[i] > 0) {
            double p = (double)counts[i] / total;
            gini -= p * p;
        }
    }
    return gini;
}

static double calc_entropy_from_counts(const int *counts, int num_classes, int total) {
    if (total == 0) return 0.0;
    double entropy = 0.0;
    for (int i = 0; i < num_classes; i++) {
        if (counts[i] > 0) {
            double p = (double)counts[i] / total;
            entropy -= p * log2(p);
        }
    }
    return entropy;
}

static double calc_mse_from_sums(double sum, double sq_sum, int total) {
    if (total == 0) return 0.0;
    double mean = sum / total;
    double variance = (sq_sum / total) - (mean * mean);
    return variance > 0 ? variance : 0.0;
}

/**
 * @brief Поиск лучшего сплита для задачи классификации (Gini / Entropy).
 */
static void find_best_split_classification(
    const Dataset *d, Subset *s, int f_idx, Criterion crit, int n_classes,
    double cur_imp, double *best_drop, int *best_f, double *best_t) 
{
    SortItem *items = (SortItem*)malloc(s->size * sizeof(SortItem));
    for (int i = 0; i < s->size; i++) {
        items[i].value = d->features[s->indices[i]][f_idx];
        items[i].index = s->indices[i];
    }
    qsort(items, s->size, sizeof(SortItem), compare_items);

    int *left_counts = (int*)calloc(n_classes, sizeof(int));
    int *right_counts = (int*)calloc(n_classes, sizeof(int));
    
    // Инициализация правой гистограммы всеми элементами узла
    for (int i = 0; i < s->size; i++) {
        right_counts[(int)d->targets[items[i].index]]++;
    }

    int left_size = 0, right_size = s->size;
    
    for (int i = 0; i < s->size - 1; i++) {
        int c = (int)d->targets[items[i].index];
        left_counts[c]++;
        right_counts[c]--;
        left_size++; 
        right_size--;

        if (items[i].value == items[i+1].value) continue; // Пропуск дубликатов

        double imp_left = (crit == CRITERION_GINI) ? 
                          calc_gini_from_counts(left_counts, n_classes, left_size) : 
                          calc_entropy_from_counts(left_counts, n_classes, left_size);
        
        double imp_right = (crit == CRITERION_GINI) ? 
                           calc_gini_from_counts(right_counts, n_classes, right_size) : 
                           calc_entropy_from_counts(right_counts, n_classes, right_size);

        double weighted_imp = (imp_left * left_size + imp_right * right_size) / s->size;
        double gain = cur_imp - weighted_imp;

        if (gain > *best_drop) {
            *best_drop = gain;
            *best_f = f_idx;
            *best_t = (items[i].value + items[i+1].value) / 2.0;
        }
    }

    free(left_counts);
    free(right_counts);
    free(items);
}

/**
 * @brief Поиск лучшего сплита для задачи регрессии (MSE).
 */
static void find_best_split_regression(
    const Dataset *d, Subset *s, int f_idx,
    double cur_imp, double *best_drop, int *best_f, double *best_t) 
{
    SortItem *items = (SortItem*)malloc(s->size * sizeof(SortItem));
    for (int i = 0; i < s->size; i++) {
        items[i].value = d->features[s->indices[i]][f_idx];
        items[i].index = s->indices[i];
    }
    qsort(items, s->size, sizeof(SortItem), compare_items);

    double left_sum = 0, left_sq_sum = 0;
    double right_sum = 0, right_sq_sum = 0;

    // Изначально всё в правой ветви
    for (int i = 0; i < s->size; i++) {
        double v = d->targets[items[i].index];
        right_sum += v;
        right_sq_sum += v * v;
    }

    int left_size = 0, right_size = s->size;
    
    for (int i = 0; i < s->size - 1; i++) {
        double v = d->targets[items[i].index];
        left_sum += v; left_sq_sum += v * v;
        right_sum -= v; right_sq_sum -= v * v;
        left_size++; right_size--;

        if (items[i].value == items[i+1].value) continue;

        double imp_left = calc_mse_from_sums(left_sum, left_sq_sum, left_size);
        double imp_right = calc_mse_from_sums(right_sum, right_sq_sum, right_size);

        double weighted_imp = (imp_left * left_size + imp_right * right_size) / s->size;
        double gain = cur_imp - weighted_imp;

        if (gain > *best_drop) {
            *best_drop = gain;
            *best_f = f_idx;
            *best_t = (items[i].value + items[i+1].value) / 2.0;
        }
    }

    free(items);
}

static double get_leaf_prediction(const Dataset *d, Subset *s, TaskType task, int n_classes) {
    if (task == TASK_REGRESSION) {
        double sum = 0;
        for (int i = 0; i < s->size; i++) sum += d->targets[s->indices[i]];
        return sum / s->size;
    } else {
        int *counts = (int*)calloc(n_classes, sizeof(int));
        for (int i = 0; i < s->size; i++) counts[(int)d->targets[s->indices[i]]]++;
        int best_class = 0, max_count = -1;
        for (int i = 0; i < n_classes; i++) {
            if (counts[i] > max_count) {
                max_count = counts[i];
                best_class = i;
            }
        }
        free(counts);
        return (double)best_class;
    }
}

static TreeNode* build_tree_recursive(TreeModel *model, const Dataset *d, Subset *s, int depth) {
    TreeNode *node = (TreeNode*)calloc(1, sizeof(TreeNode));
    node->num_samples = s->size;

    // Вычисляем примесь (Impurity) текущего узла
    if (model->task == TASK_REGRESSION) {
        double sum = 0, sq_sum = 0;
        for (int i = 0; i < s->size; i++) {
            double v = d->targets[s->indices[i]];
            sum += v; sq_sum += v * v;
        }
        node->impurity = calc_mse_from_sums(sum, sq_sum, s->size);
    } else {
        int *counts = (int*)calloc(model->n_classes, sizeof(int));
        for (int i = 0; i < s->size; i++) counts[(int)d->targets[s->indices[i]]]++;
        node->impurity = (model->criterion == CRITERION_GINI) ? 
                         calc_gini_from_counts(counts, model->n_classes, s->size) : 
                         calc_entropy_from_counts(counts, model->n_classes, s->size);
        free(counts);
    }

    // Базовые условия остановки
    if (depth >= model->max_depth || s->size < model->min_samples_split || node->impurity < EPSILON) {
        node->is_leaf = true;
        node->predicted_value = get_leaf_prediction(d, s, model->task, model->n_classes);
        return node;
    }

    int best_f = -1;
    double best_t = 0.0;
    double best_drop = -1.0;

    // Перебор признаков
    for (int f = 0; f < d->num_cols; f++) {
        if (model->task == TASK_CLASSIFICATION) {
            find_best_split_classification(d, s, f, model->criterion, model->n_classes, 
                                           node->impurity, &best_drop, &best_f, &best_t);
        } else {
            find_best_split_regression(d, s, f, node->impurity, &best_drop, &best_f, &best_t);
        }
    }

    // Если не удалось улучшить узел - делаем листом
    if (best_f == -1 || best_drop <= 0) {
        node->is_leaf = true;
        node->predicted_value = get_leaf_prediction(d, s, model->task, model->n_classes);
        return node;
    }

    // Обновляем важность признака
    model->feature_importances[best_f] += (best_drop * s->size) / d->num_rows;

    node->feature_index = best_f;
    node->threshold = best_t;

    // Физическое разделение массива индексов
    Subset sl = { (int*)malloc(s->size * sizeof(int)), 0 };
    Subset sr = { (int*)malloc(s->size * sizeof(int)), 0 };

    for (int i = 0; i < s->size; i++) {
        int idx = s->indices[i];
        if (d->features[idx][best_f] <= best_t) sl.indices[sl.size++] = idx;
        else sr.indices[sr.size++] = idx;
    }

    node->left = build_tree_recursive(model, d, &sl, depth + 1);
    node->right = build_tree_recursive(model, d, &sr, depth + 1);

    free(sl.indices);
    free(sr.indices);
    return node;
}

TreeModel* build_tree_model(Dataset *data, int max_depth, int min_samples, TaskType task, Criterion crit) {
    if (!data || data->num_rows == 0) return NULL;

    TreeModel *model = (TreeModel*)calloc(1, sizeof(TreeModel));
    model->task = task;
    model->criterion = crit;
    model->max_depth = max_depth;
    model->min_samples_split = min_samples;
    model->n_features = data->num_cols;
    model->feature_importances = (double*)calloc(model->n_features, sizeof(double));

    if (task == TASK_CLASSIFICATION) {
        int max_c = 0;
        for (int i = 0; i < data->num_rows; i++) {
            if ((int)data->targets[i] > max_c) max_c = (int)data->targets[i];
        }
        model->n_classes = max_c + 1;
    }

    Subset root_sub = { (int*)malloc(data->num_rows * sizeof(int)), data->num_rows };
    for (int i = 0; i < data->num_rows; i++) root_sub.indices[i] = i;

    model->root = build_tree_recursive(model, data, &root_sub, 0);
    free(root_sub.indices);

    // Нормализация Feature Importances (сумма = 1.0)
    double sum_imp = 0;
    for (int i = 0; i < model->n_features; i++) sum_imp += model->feature_importances[i];
    if (sum_imp > EPSILON) {
        for (int i = 0; i < model->n_features; i++) model->feature_importances[i] /= sum_imp;
    }

    return model;
}

double predict(const TreeModel *model, const double *features) {
    const TreeNode *node = model->root;
    while (!node->is_leaf) {
        if (features[node->feature_index] <= node->threshold) node = node->left;
        else node = node->right;
    }
    return node->predicted_value;
}
void print_feature_importances(const TreeModel *model) {
    printf("\n--- Важность признаков (Feature Importances) ---\n");
    for (int i = 0; i < model->n_features; i++) {
        printf("Признак X[%d] : %6.2f%%\n", i, model->feature_importances[i] * 100.0);
    }
    printf("------------------------------------------------\n");
}

void print_classification_report(const TreeModel *model, const Dataset *data) {
    int correct = 0;
    int **cm = (int**)malloc(model->n_classes * sizeof(int*));
    for (int i = 0; i < model->n_classes; i++) cm[i] = (int*)calloc(model->n_classes, sizeof(int));

    // Для бинарной классификации
    int tp = 0, fp = 0, fn = 0, tn = 0;

    for (int i = 0; i < data->num_rows; i++) {
        int act = (int)data->targets[i];
        int pred = (int)predict(model, data->features[i]);
        if (act == pred) correct++;
        cm[act][pred]++;

        if (model->n_classes == 2) {
            if (act == 1 && pred == 1) tp++;
            if (act == 0 && pred == 1) fp++;
            if (act == 1 && pred == 0) fn++;
            if (act == 0 && pred == 0) tn++;
        }
    }

    printf("\n=== ОТЧЕТ ПО КЛАССИФИКАЦИИ ===\n");
    printf("Точность (Accuracy): %.2f%%\n", ((double)correct / data->num_rows) * 100);

    if (model->n_classes == 2) {
        double precision = (tp + fp == 0) ? 0 : (double)tp / (tp + fp);
        double recall = (tp + fn == 0) ? 0 : (double)tp / (tp + fn);
        double f1 = (precision + recall == 0) ? 0 : 2.0 * (precision * recall) / (precision + recall);
        printf("Precision: %.4f | Recall: %.4f | F1-Score: %.4f\n", precision, recall, f1);
    }

    printf("\nМатрица Ошибок (Confusion Matrix):\n    ");
    for (int i = 0; i < model->n_classes; i++) printf(" P%d ", i);
    printf("\n");
    for (int i = 0; i < model->n_classes; i++) {
        printf(" A%d ", i);
        for (int j = 0; j < model->n_classes; j++) printf("%4d", cm[i][j]);
        printf("\n");
        free(cm[i]);
    }
    free(cm);
}

void print_regression_report(const TreeModel *model, const Dataset *data) {
    double mse = 0, mae = 0, rss = 0, tss = 0, mean_y = 0;

    for (int i = 0; i < data->num_rows; i++) mean_y += data->targets[i];
    mean_y /= data->num_rows;

    for (int i = 0; i < data->num_rows; i++) {
        double act = data->targets[i];
        double pred = predict(model, data->features[i]);
        double diff = act - pred;
        
        mse += diff * diff;
        mae += fabs(diff);
        rss += diff * diff;
        tss += (act - mean_y) * (act - mean_y);
    }

    mse /= data->num_rows;
    mae /= data->num_rows;
    double r2 = (tss == 0) ? 1.0 : (1.0 - (rss / tss));

    printf("\n=== ОТЧЕТ ПО РЕГРЕССИИ ===\n");
    printf("MSE (Mean Squared Error) : %.4f\n", mse);
    printf("RMSE (Root Mean Squared) : %.4f\n", sqrt(mse));
    printf("MAE (Mean Absolute Error): %.4f\n", mae);
    printf("R^2 Score (Детерминация) : %.4f\n", r2);
}

static void free_tree_node(TreeNode *node) {
    if (!node) return;
    free_tree_node(node->left);
    free_tree_node(node->right);
    free(node);
}

void free_tree_model(TreeModel *model) {
    if (!model) return;
    free_tree_node(model->root);
    free(model->feature_importances);
    free(model);
}
