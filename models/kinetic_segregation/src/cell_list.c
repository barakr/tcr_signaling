#include "cell_list.h"
#include <stdlib.h>
#include <string.h>

void cell_list_init(CellList *cl, int capacity, double r_cut, double patch_size) {
    int nc = (int)(patch_size / r_cut);
    if (nc < 1) nc = 1;
    cl->nc = nc;
    cl->capacity = capacity;
    cl->cell_size = patch_size / (double)nc;
    cl->inv_cell_size = (double)nc / patch_size;
    cl->patch_size = patch_size;
    cl->head = (int *)malloc(nc * nc * sizeof(int));
    cl->next = (int *)malloc(capacity * sizeof(int));
}

void cell_list_build(CellList *cl, const double *pos, int n_mol) {
    int nc2 = cl->nc * cl->nc;
    memset(cl->head, -1, nc2 * sizeof(int));
    for (int i = 0; i < n_mol; i++) {
        int c = cell_list_cell(cl, pos[2 * i], pos[2 * i + 1]);
        cl->next[i] = cl->head[c];
        cl->head[c] = i;
    }
}

void cell_list_move(CellList *cl, int idx, int old_cell, int new_cell) {
    if (old_cell == new_cell) return;

    /* Remove from old cell's linked list. */
    int *p = &cl->head[old_cell];
    while (*p != -1) {
        if (*p == idx) {
            *p = cl->next[idx];
            break;
        }
        p = &cl->next[*p];
    }

    /* Insert into new cell. */
    cl->next[idx] = cl->head[new_cell];
    cl->head[new_cell] = idx;
}

void cell_list_free(CellList *cl) {
    free(cl->head);
    free(cl->next);
    cl->head = NULL;
    cl->next = NULL;
}
