#include <R.h>
#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

/* FIXME: 
   Check these declarations against the C/Fortran source code.
*/

/* .Call calls */
extern SEXP _MCPanel_mcnnm(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _MCPanel_mcnnm_cv(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _MCPanel_mcnnm_fit(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _MCPanel_mcnnm_lam_range(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _MCPanel_mcnnm_wc(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _MCPanel_mcnnm_wc_cv(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _MCPanel_mcnnm_wc_fit(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _MCPanel_mcnnm_wc_lam_range(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);

static const R_CallMethodDef CallEntries[] = {
    {"_MCPanel_mcnnm",              (DL_FUNC) &_MCPanel_mcnnm,              10},
    {"_MCPanel_mcnnm_cv",           (DL_FUNC) &_MCPanel_mcnnm_cv,           11},
    {"_MCPanel_mcnnm_fit",          (DL_FUNC) &_MCPanel_mcnnm_fit,           9},
    {"_MCPanel_mcnnm_lam_range",    (DL_FUNC) &_MCPanel_mcnnm_lam_range,     7},
    {"_MCPanel_mcnnm_wc",           (DL_FUNC) &_MCPanel_mcnnm_wc,           12},
    {"_MCPanel_mcnnm_wc_cv",        (DL_FUNC) &_MCPanel_mcnnm_wc_cv,        14},
    {"_MCPanel_mcnnm_wc_fit",       (DL_FUNC) &_MCPanel_mcnnm_wc_fit,       16},
    {"_MCPanel_mcnnm_wc_lam_range", (DL_FUNC) &_MCPanel_mcnnm_wc_lam_range, 9},
    {NULL, NULL, 0}
};

void R_init_MCPanel(DllInfo *dll)
{
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
