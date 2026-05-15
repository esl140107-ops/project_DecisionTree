#ifndef DATASET_H
#define DATASET_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @struct Dataset
 * @brief Структура для хранения выборки данных (признаки и ответы).
 */
typedef struct {
    double **features; /**< Матрица признаков (X) */
    double *targets;   /**< Вектор ответов (y) */
    int num_rows;      /**< Количество строк (объектов) */
    int num_cols;      /**< Количество столбцов (признаков) */
} Dataset;

Dataset* load_csv(const char *filename, int has_header);
void train_test_split(const Dataset *source, Dataset **train, Dataset **test, double train_ratio); //разделение данных на обучающую и тестовую 
void free_dataset(Dataset *data);

#endif // DATASET_H
