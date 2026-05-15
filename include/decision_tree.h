#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include "dataset.h"
#include <stdbool.h>

/**
 * @brief Перечисления для конфигурации модели
 */
typedef enum { TASK_CLASSIFICATION, TASK_REGRESSION } TaskType;
typedef enum { CRITERION_GINI, CRITERION_ENTROPY, CRITERION_MSE } Criterion;

/**
 * @struct TreeNode
 * @brief Структура узла дерева (рекурсивная)
 */
typedef struct TreeNode {
    int feature_index;  //признак разделения
    double threshold;  //знач для разделения
    struct TreeNode *left;
    struct TreeNode *right;
    bool is_leaf;
    double predicted_value;
    int num_samples;  //колво объектов в узле
    double impurity;
} TreeNode;

/**
 * @struct TreeModel
 * @brief Обертка над деревом, хранящая метаданные и метрики важности признаков.
 */
typedef struct {
    TreeNode *root;
    TaskType task;  //тип задачи(классификация/регрессия)
    Criterion criterion;
    int max_depth;
    int min_samples_split;
    int n_features; //колво признаков данных
    int n_classes;               /**< Используется только для классификации */
    double *feature_importances; /**< Массив важности признаков */
} TreeModel;

TreeModel* build_tree_model(Dataset *data, int max_depth, int min_samples, TaskType task, Criterion crit);
double predict(const TreeModel *model, const double *features);  //предсказ знач для объекта

void print_classification_report(const TreeModel *model, const Dataset *data);
void print_regression_report(const TreeModel *model, const Dataset *data);
void print_feature_importances(const TreeModel *model);

void free_tree_model(TreeModel *model);

#endif // DECISION_TREE_H
