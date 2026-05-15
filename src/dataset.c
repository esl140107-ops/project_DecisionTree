#include "dataset.h"
#include <time.h>

Dataset* load_csv(const char *filename, int has_header) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "[Error] Не удалось открыть файл %s\n", filename);
        return NULL;
    }

    char line[8192];
    int rows = 0;  //счётчик строк
    int cols = 0;  //столбцов

    //подсчет размерности
    while (fgets(line, sizeof(line), file)) {
        if (rows == 0) {
            char *tmp = strdup(line);  //копия строки
            char *tok = strtok(tmp, ",\n");  //разбиение на токены
            while (tok) { cols++; tok = strtok(NULL, ",\n"); }
            free(tmp);
        }
        rows++;
    }

    if (has_header) rows--;
    cols--; // Последняя колонка - целевая

    if (rows <= 0 || cols <= 0) {
        fclose(file);
        return NULL;
    }

    Dataset *d = (Dataset*)malloc(sizeof(Dataset));
    d->num_rows = rows;
    d->num_cols = cols;
    d->features = (double**)malloc(rows * sizeof(double*));
    d->targets = (double*)malloc(rows * sizeof(double));

    rewind(file);
    if (has_header) fgets(line, sizeof(line), file);

    for (int i = 0; i < rows; i++) {
        if (!fgets(line, sizeof(line), file)) break;
        d->features[i] = (double*)malloc(cols * sizeof(double));
        
        char *ptr = line;
        for (int j = 0; j < cols; j++) {
            d->features[i][j] = strtod(ptr, &ptr);
            if (*ptr == ',') ptr++;
        }
        d->targets[i] = strtod(ptr, NULL);
    }

    fclose(file);
    return d;
}
//разделение на обучающую и тестовую 
void train_test_split(const Dataset *source, Dataset **train, Dataset **test, double train_ratio) {  
    if (!source || !train || !test) return;

    int train_size = (int)(source->num_rows * train_ratio);
    int test_size = source->num_rows - train_size;

    *train = (Dataset*)malloc(sizeof(Dataset));
    (*train)->num_rows = train_size;
    (*train)->num_cols = source->num_cols;
    (*train)->features = (double**)malloc(train_size * sizeof(double*));
    (*train)->targets = (double*)malloc(train_size * sizeof(double));

    *test = (Dataset*)malloc(sizeof(Dataset));
    (*test)->num_rows = test_size;
    (*test)->num_cols = source->num_cols;
    (*test)->features = (double**)malloc(test_size * sizeof(double*));
    (*test)->targets = (double*)malloc(test_size * sizeof(double));

    int *indices = (int*)malloc(source->num_rows * sizeof(int));
    for (int i = 0; i < source->num_rows; i++) indices[i] = i;

    //алгоритм Фишера-Йейтса
    srand((unsigned int)time(NULL));  //генерация случ чисел
    for (int i = source->num_rows - 1; i > 0; i--) {  //получаем случайный индекс
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }

    //распределение данных
    for (int i = 0; i < train_size; i++) {
        int src_idx = indices[i];
        (*train)->features[i] = (double*)malloc(source->num_cols * sizeof(double));
        memcpy((*train)->features[i], source->features[src_idx], source->num_cols * sizeof(double));
        (*train)->targets[i] = source->targets[src_idx];
    }

    for (int i = 0; i < test_size; i++) {
        int src_idx = indices[train_size + i];
        (*test)->features[i] = (double*)malloc(source->num_cols * sizeof(double));
        memcpy((*test)->features[i], source->features[src_idx], source->num_cols * sizeof(double));
        (*test)->targets[i] = source->targets[src_idx];
    }

    free(indices);
}

void free_dataset(Dataset *data) {
    if (!data) return;
    for (int i = 0; i < data->num_rows; i++) {
        free(data->features[i]);
    }
    free(data->features);
    free(data->targets);
    free(data);
}
