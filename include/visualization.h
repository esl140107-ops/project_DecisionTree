#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include "decision_tree.h"

void export_tree_to_dot(const TreeModel *model, const char *filename);
void print_tree_console(const TreeModel *model);

#endif // VISUALIZATION_H
