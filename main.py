# =================================  TESTY  ===================================
# Testy do tego pliku zostały podzielone na dwie kategorie:
#
#  1. `..._invalid_input`:
#     - Sprawdzające poprawną obsługę nieprawidłowych danych wejściowych.
#
#  2. `..._correct_solution`:
#     - Weryfikujące poprawność wyników dla prawidłowych danych wejściowych.
# =============================================================================
import numpy as np
import numpy.polynomial.polynomial as nppoly


def roots_20(coef: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Funkcja wyznaczająca miejsca zerowe wielomianu funkcją
    `nppoly.polyroots()`, najpierw lekko zaburzając wejściowe współczynniki 
    wielomianu (N(0,1) * 1e-10).

    Args:
        coef (np.ndarray): Wektor współczynników wielomianu (n,).

    Returns:
        (tuple[np.ndarray, np. ndarray]):
            - Zaburzony wektor współczynników (n,),
            - Wektor miejsc zerowych (m,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(coef, np.ndarray):
        return None
    if coef.ndim != 1:
        return None
    n = len(coef)
    zaburzenie = np.random.random_sample(n,) * 1e-10
    coef_zaburzone = coef + zaburzenie
    roots = nppoly.polyroots(coef_zaburzone)
    return coef_zaburzone, roots


def frob_a(coef: np.ndarray) -> np.ndarray | None:
    if not isinstance(coef, np.ndarray):
        return None
    if coef.ndim != 1:
        return None

    if len(coef) < 2:
        return None

    n = len(coef) - 1
    F = np.zeros((n, n))

    for i in range(n - 1):
        F[i, i + 1] = 1

    for j in range(n):
        F[n - 1, j] = -coef[j] / coef[-1]

    return F
    

def is_nonsingular(A: np.ndarray) -> bool | None:
    """Funkcja sprawdzająca czy podana macierz NIE JEST singularna. Przy
    implementacji należy pamiętać o definicji zera maszynowego.

    Args:
        A (np.ndarray): Macierz (n,n) do przetestowania.

    Returns:
        (bool): `True`, jeżeli macierz A nie jest singularna, w przeciwnym 
            wypadku `False`.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(A, np.ndarray):
        return None
    if A.ndim != 2:
        return None
    if A.shape[0] != A.shape[1]:
        return None
    
    if np.linalg.det(A) == 0:
        return False
    else:
        return True

