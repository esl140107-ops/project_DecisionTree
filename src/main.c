#include <stdio.h>
#include <stdlib.h>
#include "dataset.h"
#include "decision_tree.h"
#include "visualization.h"
#include <windows.h> 

int main(int argc, char **argv) {
    //включаем поддержку кодировки UTF-8 в консоли Windows
    SetConsoleOutputCP(CP_UTF8);

    if (argc < 4) {
        printf("Использование: %s <csv> <глубина> <задача c/r> [критерий g/e/m]\n", argv[0]);
        printf("Пример (Классификация, Gini):  %s data.csv 5 c g\n", argv[0]);
        printf("Пример (Регрессия, MSE):       %s reg.csv 10 r m\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *csv_path = argv[1];
    int max_depth = atoi(argv[2]);
    char task_char = argv[3][0];
    char crit_char = (argc > 4) ? argv[4][0] : 'g';

    TaskType task = (task_char == 'r') ? TASK_REGRESSION : TASK_CLASSIFICATION;
    Criterion crit = CRITERION_GINI;
    if (crit_char == 'e') crit = CRITERION_ENTROPY;
    if (crit_char == 'm' || task == TASK_REGRESSION) crit = CRITERION_MSE;

    printf("[System] Загрузка данных из %s...\n", csv_path);
    Dataset *data = load_csv(csv_path, 1);
    if (!data) return EXIT_FAILURE;

    printf("[System] Объектов: %d, Признаков: %d\n", data->num_rows, data->num_cols);

    Dataset *train, *test;
    train_test_split(data, &train, &test, 0.8);

    printf("[System] Обучение модели CART...\n");
    TreeModel *model = build_tree_model(train, max_depth, 2, task, crit);

    print_feature_importances(model);
    print_tree_console(model);

    if (task == TASK_CLASSIFICATION) print_classification_report(model, test);
    else print_regression_report(model, test);

    export_tree_to_dot(model, "tree.dot");

    free_tree_model(model);
    free_dataset(train);
    free_dataset(test);
    free_dataset(data);

    return EXIT_SUCCESS;
}
