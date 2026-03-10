#ifndef CELL_LIST_H
#define CELL_LIST_H

/**
 * Linked-list cell list for O(N) short-range neighbor queries on a periodic
 * 2D domain.  Cell size is set to r_cut so that only the 3x3 neighborhood
 * needs to be searched.
 */
typedef struct CellList {
    int *head;            /* [nc * nc] first molecule index in cell, or -1 */
    int *next;            /* [capacity] linked-list next pointer, -1 = end */
    int nc;               /* cells per dimension */
    int capacity;         /* max molecules (size of next array) */
    double cell_size;
    double inv_cell_size;
    double patch_size;
} CellList;

/** Allocate internal arrays.  Must call cell_list_free when done. */
void cell_list_init(CellList *cl, int capacity, double r_cut, double patch_size);

/** Rebuild from scratch using current positions. */
void cell_list_build(CellList *cl, const double *pos, int n_mol);

/** Return flat cell index for a position (x, y). */
static inline int cell_list_cell(const CellList *cl, double x, double y) {
    int ci = (int)(x * cl->inv_cell_size);
    int cj = (int)(y * cl->inv_cell_size);
    if (ci < 0) ci = 0;
    if (ci >= cl->nc) ci = cl->nc - 1;
    if (cj < 0) cj = 0;
    if (cj >= cl->nc) cj = cl->nc - 1;
    return ci * cl->nc + cj;
}

/** Move molecule idx from old_cell to new_cell (O(1) amortized). */
void cell_list_move(CellList *cl, int idx, int old_cell, int new_cell);

/** Free internal arrays. */
void cell_list_free(CellList *cl);

#endif /* CELL_LIST_H */
