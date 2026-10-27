import ast
import json
from decimal import Decimal, localcontext
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import random
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, log_expit
from scipy import stats


SCRIPT_FILES = [
    'MCMC_all_pairs.py',
    'MCMC_single_pair.py',
    'MCMC_single_system_all_pairs.py',
    'MCMC_planet_masses.py',
]

FILE_OPERATION_SCRIPT_FILES = [
    'MCMC_all_pairs.py',
    'MCMC_single_pair.py',
    'MCMC_single_system_all_pairs.py',
    'gamma_distribution.py',
    'instrument_mass_histogram.py',
    'mass_hist_residual.py',
    'mu_tilde_distribution.py',
    'mutual_hill_radius.py',
    'orbital_seperation.py',
    'total_d.py',
    'MCMC_planet_masses.py',
]


def test_single_pair_example_notebook_uses_shared_model_and_keeps_paths_private():
    path = Path(__file__).parent / 'examples' / 'single_pair_posterior.ipynb'
    print('Reading from examples/single_pair_posterior.ipynb...')
    notebook = json.loads(path.read_text(encoding='utf-8'))
    code = '\n'.join(
        ''.join(cell['source'])
        for cell in notebook['cells']
        if cell['cell_type'] == 'code'
    )
    document = json.dumps(notebook)

    assert 'from pairwise_model import combine_log_posterior, log_prior, orbital_log_likelihood' in code
    assert 'from MCMC_single_pair import' not in code
    assert 'mass_estimation_single_pair_posterior.{typefileplot}' in code
    assert 'mass_estimation_single_pair_posterior.html' in code
    assert 'not a sampled chain' in document
    assert '/' + 'Users/' not in document


@pytest.mark.parametrize('verbosity', ['0', '1'])
@pytest.mark.parametrize('filename', SCRIPT_FILES)
def test_mcmc_catalog_loading_is_independent_of_working_directory(filename, verbosity, tmp_path, monkeypatch, capsys):
    import pandas as pd

    path = Path(__file__).with_name(filename).resolve()
    print(f'Reading from {path}...')
    tree = ast.parse(path.read_text())
    loading_end = next(
        index for index, node in enumerate(tree.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in {'datafile', 'logD'} for target in node.targets)
    )
    selected = [
        node for node in tree.body[:loading_end + 1]
        if (
            isinstance(node, ast.ImportFrom) and node.module == 'tdpy.verbosity'
        ) or (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in {'catalog_path', 'filename', 'datafile', 'logD'} for target in node.targets)
        ) or (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name) and node.value.func.id == 'print'
        )
    ]
    pairwise = filename in SCRIPT_FILES[:2]
    catalog_path = path.with_name('logD.csv' if pairwise else 'only_transit_data_28062023.csv')
    print(f'Reading from {catalog_path}...')
    expected = pd.read_csv(catalog_path, **({} if pairwise else {'skiprows': 202}))
    capsys.readouterr()
    monkeypatch.setenv('TDPY_VERBOSITY', verbosity)
    monkeypatch.chdir(tmp_path)
    namespace = {'Path': Path, '__file__': str(path), 'pd': pd}

    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)

    actual = namespace['logD' if filename == 'MCMC_all_pairs.py' else 'datafile']
    pd.testing.assert_frame_equal(actual, expected)
    expected_output = f'Reading from {catalog_path}...\n' if verbosity == '1' else ''
    assert capsys.readouterr().out == expected_output


@pytest.mark.parametrize('filename', SCRIPT_FILES)
def test_mcmc_sampler_progress_uses_tdpy_verbosity(filename):
    path = Path(__file__).with_name(filename)
    print(f'Reading from {path}...')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    progress_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'run_mcmc'
    ]

    assert progress_calls
    assert all(
        any(
            keyword.arg == 'progress'
            and isinstance(keyword.value, ast.Call)
            and isinstance(keyword.value.func, ast.Name)
            and keyword.value.func.id == 'retr_boolverb'
            for keyword in call.keywords
        )
        for call in progress_calls
    )


@pytest.mark.parametrize('filename', FILE_OPERATION_SCRIPT_FILES)
def test_file_operations_have_exact_verbosity_narration(filename):
    path = Path(__file__).with_name(filename)
    print(f'Reading from {path}...')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == 'tdpy.verbosity'
        and any(alias.name == 'print' for alias in node.names)
        for node in tree.body
    )

    def file_operation(statement):
        if not isinstance(statement, (ast.Assign, ast.Expr)):
            return False
        return any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {'read_csv', 'to_csv', 'savefig'}
            for node in ast.walk(statement)
        )

    def narration(statement):
        return (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == 'print'
            and any(
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.startswith(('Reading from ', 'Writing to '))
                for node in ast.walk(statement.value)
            )
        )

    narrated_operations = 0
    for parent in ast.walk(tree):
        body = getattr(parent, 'body', None)
        if not isinstance(body, list):
            continue
        for index, statement in enumerate(body):
            if file_operation(statement):
                assert index > 0 and narration(body[index - 1]), (filename, statement.lineno)
                narrated_operations += 1

    assert narrated_operations


@pytest.mark.parametrize('verbosity', ['0', '1'])
def test_single_pair_console_output_follows_tdpy_verbosity(verbosity, tmp_path, monkeypatch, capsys):
    path = Path(__file__).with_name('MCMC_single_pair.py').resolve()
    tree = ast.parse(path.read_text(encoding='utf-8'))
    loading_end = next(
        index for index, node in enumerate(tree.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'datafile' for target in node.targets)
    )
    selected = [
        node for node in tree.body[:loading_end + 1]
        if (isinstance(node, ast.ImportFrom) and node.module == 'tdpy.verbosity')
        or (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in {'catalog_path', 'datafile'}
            for target in node.targets
        ))
        or (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name) and node.value.func.id == 'print')
    ]
    monkeypatch.setenv('TDPY_VERBOSITY', verbosity)
    monkeypatch.chdir(tmp_path)
    namespace = {'Path': Path, '__file__': str(path), 'pd': pd}

    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)

    expected_output = f'Reading from {path.with_name("logD.csv")}...\n' if verbosity == '1' else ''
    assert capsys.readouterr().out == expected_output
    progress_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'run_mcmc'
    ]
    assert len(progress_calls) == 2
    assert all(
        isinstance(keyword.value, ast.Call)
        and isinstance(keyword.value.func, ast.Name)
        and keyword.value.func.id == 'retr_boolverb'
        for call in progress_calls
        for keyword in call.keywords if keyword.arg == 'progress'
    )


def numerical_helpers(filename):
    """Load numerical helpers without reading catalogs or starting MCMC chains."""
    path = Path(__file__).with_name(filename)
    print(f'Reading from {path}...')
    tree = ast.parse(path.read_text())
    imports = [
        node for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module in {'scipy.special', 'tdpy', 'pairwise_model'}
    ]
    definitions = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in {'log_K', 'log_prior', 'log_likelihood', 'log_posterior'}
    ]
    namespace = {
        'np': np, 'scp': stats, 'logD': 0.0, 'logD_value': 0.0,
        'logD_array': np.array([0.0]), 'j': 0,
    }
    exec(compile(ast.Module(body=imports + definitions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def catalog_preparation(filename):
    path = Path(__file__).with_name(filename)
    print(f'Reading from {path}...')
    tree = ast.parse(path.read_text())
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == 'pairwise_model']
    loop = next(node for node in tree.body if isinstance(node, ast.While) and any(isinstance(part, ast.Name) and part.id == 'number_of_systems' for part in ast.walk(node.test)))
    end = next(index for index, node in enumerate(loop.body) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'logD_array' for target in node.targets))
    return compile(ast.Module(body=imports + loop.body[:end + 1], type_ignores=[]), str(path), 'exec')


@pytest.mark.parametrize('filename', SCRIPT_FILES[2:])
@pytest.mark.parametrize('custom_index', [False, True])
@pytest.mark.parametrize('nullable_periods', [False, True])
@pytest.mark.parametrize(('hosts', 'periods', 'position', 'ratios'), [
    (['A'], [2.0], 0, []),
    (['A', 'B', 'B'], [8.0, 2.0, 4.0], 1, [2.0]),
    (['A', 'A', 'B', 'B', 'B'], [1.0, 3.0, 8.0, 4.0, 2.0], 2, [2.0, 2.0]),
    (['A', 'A', 'B', 'B', 'C'], [1.0, 2.0, 3.0, 9.0, 15.0], 2, [3.0]),
    (['A', 'A', 'B'], [1.0, 4.0, 10.0], 0, [4.0]),
])
def test_multi_system_catalog_preparation_uses_shared_bounded_positional_pairs(filename, custom_index, nullable_periods, hosts, periods, position, ratios):
    import pandas as pd
    import pairwise_model
    from astropy import constants

    stellar_masses = np.linspace(1.0, 2.0, len(hosts))  # [M_sun]
    catalog = pd.DataFrame({'hostname': hosts, 'pl_orbper': periods, 'st_mass': stellar_masses}, index=np.arange(len(hosts)) * 2 + 11 if custom_index else None)
    if nullable_periods:
        catalog['pl_orbper'] = catalog['pl_orbper'].astype('Float64')
    namespace = {'np': np, 'ast': constants, 'datafile': catalog, 'i': position, 'row_counter': position}

    with np.errstate(all='raise'):
        exec(catalog_preparation(filename), namespace)

    assert namespace['adjacent_orbital_separations'] is pairwise_model.adjacent_orbital_separations
    expected = np.array([2 * ((ratio ** (2 / 3) - 1) / (ratio ** (2 / 3) + 1)) for ratio in ratios])
    np.testing.assert_array_equal(namespace['separations'], expected)
    logarithm = np.log if filename == 'MCMC_single_system_all_pairs.py' else np.log10
    np.testing.assert_array_equal(namespace['logD_array'], logarithm(expected))
    assert namespace['row_number'] == len(ratios) + 1
    if filename == 'MCMC_planet_masses.py':
        expected_mass = stellar_masses[position] * constants.M_sun.value / constants.M_earth.value  # [M_earth]
        assert namespace['st_mass'] == expected_mass


@pytest.mark.parametrize('filename', SCRIPT_FILES[2:])
@pytest.mark.parametrize('custom_index', [False, True])
@pytest.mark.parametrize(('hosts', 'position'), [
    (['A', 'A', 'B', 'B', 'A', 'A'], 0),
    (['A', 'B', 'A'], 0),
    (['A', 'A', 'B'], 1),
    (['A', 'B', 'B'], 2),
    (['A', 'A', 'A'], 2),
])
def test_multi_system_catalog_preparation_rejects_split_or_partial_host_blocks(filename, custom_index, hosts, position):
    import pandas as pd
    from astropy import constants

    catalog = pd.DataFrame({'hostname': hosts, 'pl_orbper': np.arange(1.0, len(hosts) + 1.0), 'st_mass': np.ones(len(hosts))}, index=np.arange(len(hosts)) * 2 + 11 if custom_index else None)
    namespace = {'np': np, 'ast': constants, 'datafile': catalog, 'i': position, 'row_counter': position}

    with np.errstate(all='raise'), pytest.raises(ValueError, match='host rows must be contiguous'):
        exec(catalog_preparation(filename), namespace)

    assert 'logD_array' not in namespace


@pytest.mark.parametrize('filename', SCRIPT_FILES[2:])
@pytest.mark.parametrize('dtype', ['float64', 'Float64'])
@pytest.mark.parametrize('invalid_period', [0.0, -1.0, np.nan, np.inf, -np.inf, None])
def test_multi_system_catalog_preparation_rejects_invalid_paired_periods(filename, dtype, invalid_period):
    import pandas as pd
    from astropy import constants

    catalog = pd.DataFrame({'hostname': ['B', 'A', 'A'], 'pl_orbper': pd.Series([np.nan, invalid_period, 2.0], dtype=dtype), 'st_mass': [1.0, 1.0, 1.0]})
    namespace = {'np': np, 'ast': constants, 'datafile': catalog, 'i': 1, 'row_counter': 1}

    with np.errstate(all='raise'), pytest.raises(ValueError, match='periods must be finite and positive'):
        exec(catalog_preparation(filename), namespace)

    assert 'logD_array' not in namespace


@pytest.mark.parametrize('filename', SCRIPT_FILES[2:])
@pytest.mark.parametrize('dtype', ['object', 'string'])
@pytest.mark.parametrize('hostname', ['', ' \t', None, np.nan, pd.NA])
def test_multi_system_catalog_preparation_rejects_missing_or_blank_hosts(filename, dtype, hostname):
    import pandas as pd
    from astropy import constants

    catalog = pd.DataFrame({'hostname': pd.Series(['B', hostname, hostname], dtype=dtype), 'pl_orbper': [1.0, 2.0, 4.0], 'st_mass': [1.0, 1.0, 1.0]})
    original = catalog.copy(deep=True)
    namespace = {'np': np, 'ast': constants, 'datafile': catalog, 'i': 1, 'row_counter': 1}

    with np.errstate(all='raise'), pytest.raises(ValueError, match='nonmissing, nonblank identifier'):
        exec(catalog_preparation(filename), namespace)

    assert 'logD_array' not in namespace
    pd.testing.assert_frame_equal(catalog, original)


@pytest.mark.parametrize('filename', SCRIPT_FILES[2:])
@pytest.mark.parametrize('dtype', ['object', 'string', 'category'])
def test_multi_system_catalog_preparation_preserves_nonblank_host_identifiers(filename, dtype):
    import pandas as pd
    from astropy import constants

    catalog = pd.DataFrame({'hostname': pd.Series([None, ' A ', ' A '], dtype=dtype), 'pl_orbper': [np.nan, 1.0, 2.0], 'st_mass': [1.0, 1.0, 1.0]})
    original = catalog.copy(deep=True)
    namespace = {'np': np, 'ast': constants, 'datafile': catalog, 'i': 1, 'row_counter': 1}

    with np.errstate(all='raise'):
        exec(catalog_preparation(filename), namespace)

    expected = 2 * ((2.0 ** (2 / 3) - 1) / (2.0 ** (2 / 3) + 1))
    np.testing.assert_array_equal(namespace['separations'], [expected])
    assert namespace['row_number'] == 2
    pd.testing.assert_frame_equal(catalog, original)


@pytest.mark.parametrize('filename', SCRIPT_FILES[2:])
@pytest.mark.parametrize('unused_period', [np.nan, 0.0])
def test_multi_system_catalog_preparation_keeps_singleton_skip_behavior(filename, unused_period):
    import pandas as pd
    from astropy import constants

    catalog = pd.DataFrame({'hostname': ['A'], 'pl_orbper': pd.Series([unused_period], dtype='Float64'), 'st_mass': [1.0]})
    namespace = {'np': np, 'ast': constants, 'datafile': catalog, 'i': 0, 'row_counter': 0}

    with np.errstate(all='raise'):
        exec(catalog_preparation(filename), namespace)

    assert namespace['separations'].shape == namespace['logD_array'].shape == (0,)
    assert namespace['row_number'] == 1


@pytest.mark.parametrize('filename', SCRIPT_FILES[2:])
@pytest.mark.parametrize(('row_count', 'position', 'processed', 'expected'), [
    (0, 0, 0, False), (1, 0, 0, True), (1, 1, 0, False),
    (3, 2, 4, True), (3, 3, 4, False), (3, 0, 5, False),
])
def test_multi_system_outer_loop_stops_at_catalog_or_requested_count(filename, row_count, position, processed, expected):
    import pandas as pd

    path = Path(__file__).with_name(filename)
    print(f'Reading from {path}...')
    tree = ast.parse(path.read_text())
    loop = next(node for node in tree.body if isinstance(node, ast.While) and any(isinstance(part, ast.Name) and part.id == 'number_of_systems' for part in ast.walk(node.test)))
    namespace = {'datafile': pd.DataFrame({'hostname': ['A'] * row_count}), 'i': position, 'row_counter': position, 'a': processed, 'number_of_systems': 5}

    actual = eval(compile(ast.Expression(body=loop.test), str(path), 'eval'), namespace)

    assert actual is expected


@pytest.mark.parametrize('base', [1e-300, 1.0, 1e300])
@pytest.mark.parametrize('ulp_width', [1, 16, 1024])
@pytest.mark.parametrize('reverse', [False, True])
def test_shared_orbital_separation_retains_narrow_period_intervals(base, ulp_width, reverse):
    import pandas as pd
    from pairwise_model import adjacent_orbital_separations

    upper = base + ulp_width * np.spacing(base)  # [day]
    periods = [upper, base] if reverse else [base, upper]
    catalog = pd.DataFrame({'hostname': ['A', 'A'], 'pl_orbper': periods})
    with localcontext() as context:
        context.prec = 90
        ratio = Decimal.from_float(upper) / Decimal.from_float(base)
        power = (ratio.ln() * Decimal(2) / Decimal(3)).exp()
        expected = float(2 * (power - 1) / (power + 1))

    with np.errstate(all='raise'):
        actual, row_count = adjacent_orbital_separations(catalog, 0)

    assert actual[0] > 0
    assert actual[0] == pytest.approx(expected, rel=2e-15)
    assert row_count == 2


@pytest.mark.parametrize('periods', [
    [1e-300, 1e300],
    [np.finfo(float).tiny, np.finfo(float).max],
    [np.nextafter(0.0, 1.0), np.finfo(float).max],
])
@pytest.mark.parametrize('reverse', [False, True])
def test_shared_orbital_separation_preserves_wide_ratio_limit_without_overflow(periods, reverse):
    import pandas as pd
    from pairwise_model import adjacent_orbital_separations

    catalog = pd.DataFrame({'hostname': ['A', 'A'], 'pl_orbper': periods[::-1] if reverse else periods})
    with np.errstate(all='raise'):
        actual, row_count = adjacent_orbital_separations(catalog, 0)

    np.testing.assert_array_equal(actual, [2.0])
    assert row_count == 2


def test_shared_orbital_separation_preserves_ordinary_calculations_exactly():
    import pandas as pd
    from pairwise_model import adjacent_orbital_separations

    for base in [1e-200, 1.0, 1e200]:
        for ratio in [1.01, 1.5, 2.0, 10.0]:
            for reverse in [False, True]:
                periods = np.array([base, base * ratio])  # [day]
                catalog = pd.DataFrame({'hostname': ['A', 'A'], 'pl_orbper': periods[::-1] if reverse else periods})
                original_ratio = periods[1] / periods[0]
                expected = 2 * ((original_ratio ** (2 / 3) - 1) / (original_ratio ** (2 / 3) + 1))
                actual, _ = adjacent_orbital_separations(catalog, 0)
                assert actual[0] == expected


@pytest.mark.parametrize(('first_period', 'second_period'), [
    (1.0, 1.01), (1.0, 1.5), (1.0, 2.0), (1.0, 10.0),
    (1.5, 1.0), (2.0, 1.0), (10.0, 1.0),
])
def test_orbital_separation_from_periods_preserves_ordinary_formula(first_period, second_period):
    from pairwise_model import orbital_separation_from_periods

    ratio = second_period / first_period
    ratio_power = ratio ** (2 / 3)
    expected = 2 * ((ratio_power - 1) / (ratio_power + 1))

    assert orbital_separation_from_periods(first_period, second_period) == expected


@pytest.mark.parametrize(('first_period', 'second_period'), [
    (1.0, np.nextafter(1.0, 2.0)),
    (np.nextafter(1.0, 2.0), 1.0),
])
def test_orbital_separation_from_periods_retains_signed_close_pairs(first_period, second_period):
    from pairwise_model import orbital_separation_from_periods

    expected_log_ratio = np.log1p((second_period - first_period) / first_period)
    expected = 2 * np.tanh(expected_log_ratio / 3)

    assert orbital_separation_from_periods(first_period, second_period) == expected


def test_orbital_separation_from_periods_preserves_overflow_limits():
    from pairwise_model import orbital_separation_from_periods

    with np.errstate(all='raise'):
        assert orbital_separation_from_periods(1e-300, 1e300) == 2.0
        assert orbital_separation_from_periods(1e300, 1e-300) == -2.0


def test_adjacent_period_spacings_preserves_host_pairs_and_instrument_filter():
    import pandas as pd
    from pairwise_model import adjacent_period_spacings

    catalog = pd.DataFrame({
        "hostname": ["A", "A", "B", "B", "C", "C"],
        "pl_orbper": [1.0, 2.0, 4.0, 2.0, 3.0, 6.0],
        "disc_instrument": ["TESS CCD Array", "Kepler", "TESS CCD Array", "TESS", "Kepler CCD Array", "Kepler"],
    }, index=[10, 20, 30, 40, 50, 60])
    expected = np.array([
        2 * ((ratio ** (2 / 3) - 1) / (ratio ** (2 / 3) + 1))
        for ratio in (2.0, 0.5, 2.0)
    ])

    positions, spacings = adjacent_period_spacings(catalog)
    tess_positions, tess_spacings = adjacent_period_spacings(
        catalog, instrument="TESS CCD Array"
    )

    np.testing.assert_array_equal(positions, [0, 2, 4])
    np.testing.assert_array_equal(spacings, expected)
    np.testing.assert_array_equal(tess_positions, [0, 2])
    np.testing.assert_array_equal(tess_spacings, expected[[0, 1]])


@pytest.mark.parametrize(("masses", "axes", "system_mass"), [
    ([1.0, 2.0], [0.1, 0.2], 333000.0),
    ([10.0, 0.5], [0.04, 0.09], 666001.5),
    ([0.2, 0.1], [1.0, 4.0], 999000.3),
])
def test_shared_mutual_hill_radius_preserves_original_formula(masses, axes, system_mass):
    from pairwise_model import mutual_hill_radius

    expected = ((masses[0] + masses[1]) / (3 * system_mass)) ** (1 / 3) * ((axes[0] + axes[1]) / 2)

    assert mutual_hill_radius(masses, axes, system_mass) == expected


def test_adjacent_mutual_hill_radii_preserves_pair_positions_and_system_masses():
    import pandas as pd
    from pairwise_model import adjacent_mutual_hill_radii

    catalog = pd.DataFrame({
        "hostname": ["A", "A", "B", "B", "A"],
        "pl_bmasse": [1.0, 2.0, 3.0, 4.0, 5.0],
        "pl_orbsmax": [0.1, 0.2, 0.3, 0.5, 0.8],
        "st_mass": [1.0, 9.0, 2.0, 8.0, 7.0],
    })
    solar_mass_in_earth_masses = 0.33261191609863575142462501642665 * 10**6
    expected = [
        ((1.0 + 2.0) / (3 * (1.0 + 2.0 + 5.0 + solar_mass_in_earth_masses))) ** (1 / 3) * ((0.1 + 0.2) / 2),
        ((3.0 + 4.0) / (3 * (3.0 + 4.0 + 2.0 * solar_mass_in_earth_masses))) ** (1 / 3) * ((0.3 + 0.5) / 2),
    ]

    positions, radii = adjacent_mutual_hill_radii(catalog)

    np.testing.assert_array_equal(positions, [0, 2])
    np.testing.assert_array_equal(radii, expected)


def test_adjacent_mutual_hill_radii_returns_integer_empty_positions():
    import pandas as pd
    from pairwise_model import adjacent_mutual_hill_radii

    catalog = pd.DataFrame({
        "hostname": ["A"],
        "pl_bmasse": [1.0],
        "pl_orbsmax": [0.1],
        "st_mass": [1.0],
    })

    positions, radii = adjacent_mutual_hill_radii(catalog)

    assert positions.dtype.kind in "iu"
    assert positions.size == radii.size == 0


@pytest.mark.parametrize(
    ("mass_location", "mass_scale"),
    [(-4.31921754762478, 0.28778378882844535), (-4.3041661629482, 0.30148352295199915)],
)
def test_initial_pairwise_positions_preserves_seeded_draw_sequence(mass_location, mass_scale):
    from pairwise_model import initial_pairwise_positions

    np.random.seed(17)
    log_mass = stats.norm.rvs(size=12, loc=mass_location, scale=mass_scale)
    gamma = []
    while len(gamma) != 12:
        candidate = stats.norm.rvs(1.0, 0.3)
        if 0 < candidate < 1:
            gamma.append(candidate)
    expected = np.transpose(np.vstack((log_mass, gamma)))

    np.random.seed(17)
    actual = initial_pairwise_positions(12, mass_location, mass_scale)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("gamma", [0.01, 0.5, 0.99, [0.01, 0.5, 0.99]])
def test_pair_mass_fractions_preserve_original_formulas(gamma):
    from pairwise_model import pair_mass_fractions

    gamma = np.asarray(gamma)
    expected_lower = gamma * (1 + gamma) ** (-1)
    expected_upper = (1 + gamma) ** (-1)

    lower, upper = pair_mass_fractions(gamma)

    np.testing.assert_array_equal(lower, expected_lower)
    np.testing.assert_array_equal(upper, expected_upper)


@pytest.mark.parametrize(
    ("masses", "stellar_mass"),
    [
        ([1.0, 2.0, 4.0], 10.0),
        (np.array([[1.0, 10.0], [2.0, 20.0], [4.0, 40.0]]), 100.0),
    ],
)
def test_adjacent_pair_mass_ratios_preserve_vector_shapes(masses, stellar_mass):
    from pairwise_model import adjacent_pair_mass_ratios

    expected = (np.asarray(masses[:-1]) + np.asarray(masses[1:])) / stellar_mass

    np.testing.assert_array_equal(
        adjacent_pair_mass_ratios(masses, stellar_mass), expected
    )


def test_log_mu_tilde_inverts_the_shared_orbital_spacing_transform():
    from pairwise_model import log_K, log_mu_tilde_from_log_K

    log_mass = np.array([-8.0, -4.0, 0.5])
    log_separation = np.array([-3.0, -1.0, 0.2])
    log_spacing = log_K(log_mass, log_separation)

    np.testing.assert_array_equal(
        log_mu_tilde_from_log_K(log_separation, log_spacing),
        3 * (log_separation - log_spacing) + np.log10(3.0),
    )
    np.testing.assert_allclose(
        log_mu_tilde_from_log_K(log_separation, log_spacing), log_mass,
        rtol=0.0, atol=2e-15,
    )


@pytest.mark.parametrize('filename', SCRIPT_FILES)
def test_sigmoid_uses_scipy_and_preserves_finite_values_and_limits(filename):
    sigmoid = numerical_helpers(filename)['sigmoid']
    assert sigmoid is expit
    coordinates = np.linspace(-10.0, 10.0, 101)
    expected = np.exp(coordinates) / (np.exp(coordinates) + 1.0)
    np.testing.assert_allclose(sigmoid(coordinates), expected, rtol=5e-15)
    with np.errstate(over='raise', invalid='raise'):
        np.testing.assert_array_equal(sigmoid([-np.inf, -1000.0, 0.0, 1000.0, np.inf]), [0.0, 0.0, 0.5, 1.0, 1.0])
    assert np.isnan(sigmoid(np.nan))


@pytest.mark.parametrize(('steps', 'walkers'), [(2, 3), (250, 100)])
@pytest.mark.parametrize('fit_count', [0, 1, 2, 11])
def test_multi_system_mass_summary_preserves_every_fitted_sample(steps, walkers, fit_count):
    import pandas as pd

    path = Path(__file__).with_name('MCMC_single_system_all_pairs.py')
    print(f'Reading from {path}...')
    tree = ast.parse(path.read_text())
    initialization = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'mass_array' for target in node.targets))
    accumulation = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'mass_array' for target in node.targets) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'vstack')
    names = {'mass_array_list', 'mass_array_merged', 'log_mass', 'categories', 'subcategories', 'data'}
    summary = [node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)]
    namespace = {'np': np, 'pd': pd, 'num_steps': steps, 'nwalkers': walkers, 'which_systems': [f'host{index}' for index in range(fit_count)], 'planet_number': list(range(1, fit_count + 1))}
    exec(compile(ast.Module(body=[initialization], type_ignores=[]), str(path), 'exec'), namespace)
    expected_rows = []
    for index in range(fit_count):
        samples = np.arange(steps * walkers, dtype=float) + index * steps * walkers + 1.0  # [M_earth]
        expected_rows.append(samples)
        namespace['masses'] = samples
        exec(compile(ast.Module(body=[accumulation], type_ignores=[]), str(path), 'exec'), namespace)

    with np.errstate(all='raise'):
        exec(compile(ast.Module(body=summary, type_ignores=[]), str(path), 'exec'), namespace)

    expected = np.concatenate(expected_rows) if expected_rows else np.array([])  # [M_earth]
    assert namespace['mass_array'].shape == (fit_count, steps * walkers)
    np.testing.assert_array_equal(namespace['mass_array_merged'], expected)
    np.testing.assert_array_equal(namespace['log_mass'], np.log10(expected))
    data = namespace['data']
    assert list(data['Category']) == namespace['which_systems']
    assert list(data['Subcategory']) == namespace['planet_number']
    for actual, row in zip(data['Value'], expected_rows):
        np.testing.assert_array_equal(actual, row)


@pytest.mark.parametrize('fit_count', [0, 1, 2])
@pytest.mark.parametrize('density_failure', [False, True])
def test_multi_system_mass_summary_renders_all_samples_or_skips_empty_runs(tmp_path, monkeypatch, fit_count, density_failure):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    path = Path(__file__).with_name('MCMC_single_system_all_pairs.py')
    print(f'Reading from {path}...')
    statements = ast.parse(path.read_text()).body
    start = next(index for index, node in enumerate(statements) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'mass_array_list' for target in node.targets))
    masses = np.arange(1.0, fit_count * 24 + 1.0).reshape(fit_count, 24)  # [M_earth]
    show = Mock()
    monkeypatch.setattr(plt, 'show', show)
    if density_failure:
        monkeypatch.setattr(Axes, 'violinplot', Mock(side_effect=ValueError('synthetic density failure')))
    namespace = {'np': np, 'pd': pd, 'plt': plt, 'mass_array': masses, 'which_systems': [f'host{index}' for index in range(fit_count)], 'planet_number': list(range(1, fit_count + 1))}
    before = set(plt.get_fignums())

    with np.errstate(all='raise'):
        exec(compile(ast.Module(body=statements[start:], type_ignores=[]), str(path), 'exec'), namespace)

    created = set(plt.get_fignums()) - before
    assert show.call_count == int(fit_count > 0)
    if fit_count == 0:
        assert created == set()
        return
    try:
        assert len(created) == 2
        figures = [plt.figure(number) for number in created]
        histogram = next(figure for figure in figures if figure.get_label() == 'Total Mass')
        assert sum(patch.get_height() for patch in histogram.axes[0].patches) == masses.size
        assert histogram.axes[0].get_xlabel() == r'$\log_{10}(M/M_\oplus)$'
        assert histogram.axes[0].get_ylabel() == 'Sample Count'
        violin = next(figure for figure in figures if figure is not histogram)
        assert [label.get_text() for label in violin.axes[0].get_yticklabels()] == namespace['which_systems']
        assert len(violin.axes[0].collections) >= fit_count * 3
        for index, figure in enumerate(figures):
            FigureCanvasAgg(figure)
            figure.canvas.draw()
            assert np.ptp(np.asarray(figure.canvas.buffer_rgba())[..., :3]) > 0
            filename = tmp_path / f'summary_{index}.png'
            print(f'Writing to {filename}...')
            figure.savefig(filename)
            print(f'Reading from {filename}...')
            image = plt.imread(filename)
            assert image.shape[0] > 100 and image.shape[1] > 100
            assert np.ptp(image[..., :3]) > 0
    finally:
        for number in created:
            plt.close(number)


@pytest.mark.parametrize('filename', SCRIPT_FILES[:3])
def test_log_likelihood_stays_finite_in_the_large_separation_tail(filename):
    helpers = numerical_helpers(filename)
    mass_ratio_log = -4.0
    separations = np.array([1.0, 4.0, 40.0, 1000.0])
    observed_log = np.log10(separations) + (1 / 3) * mass_ratio_log - (1 / 3) * np.log10(3.0)

    with np.errstate(over='raise', invalid='raise', divide='raise'):
        theta = (0.5, mass_ratio_log) if filename == 'MCMC_single_system_all_pairs.py' else (mass_ratio_log, 0.5)
        actual = helpers['log_likelihood'](theta, observed_log)

    expected = -np.logaddexp(0.0, separations - np.sqrt(12.0))
    if filename == 'MCMC_all_pairs.py':
        expected = np.sum(expected)
    if filename == 'MCMC_single_system_all_pairs.py':
        expected /= np.log(10.0)
    np.testing.assert_allclose(actual, expected, rtol=5e-15, atol=1e-14)
    assert np.isfinite(actual).all()


@pytest.mark.parametrize('filename', SCRIPT_FILES[:3])
def test_log_likelihood_preserves_scalar_results(filename):
    helpers = numerical_helpers(filename)
    theta = (0.5, -4.0) if filename == 'MCMC_single_system_all_pairs.py' else (-4.0, 0.5)
    actual = helpers['log_likelihood'](theta, -2.0)
    separation = 10.0 ** helpers['log_K'](-4.0, -2.0)
    expected = np.log(1.0 - expit(separation - np.sqrt(12.0)))
    if filename == 'MCMC_single_system_all_pairs.py':
        expected /= np.log(10.0)
    assert actual == pytest.approx(expected, rel=1e-14)
    assert np.ndim(actual) == 0


@pytest.mark.parametrize('mass_coordinate', [-5.0, -4.0, -3.0])
@pytest.mark.parametrize('vector', [False, True])
def test_multi_system_shared_likelihood_preserves_ordinary_formula_exactly(mass_coordinate, vector):
    helpers = numerical_helpers('MCMC_single_system_all_pairs.py')
    observed = np.linspace(-3.0, 1.0, 41) if vector else -2.0
    previous_log_k = observed - (1 / 3) * mass_coordinate + (1 / 3) * np.log10(3.0)
    expected = log_expit(np.sqrt(12) - 10.0 ** previous_log_k) / np.log(10.0)

    with np.errstate(all='raise'):
        actual = helpers['log_likelihood']((0.5, mass_coordinate), observed)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('mass_choice', [0, 1])
@pytest.mark.parametrize('return_directions', [False, True])
@pytest.mark.parametrize('observed', [-1.0, 2.0])
def test_planet_mass_likelihood_preserves_direction_base_and_mass_choices(mass_choice, return_directions, observed):
    helpers = numerical_helpers('MCMC_planet_masses.py')
    draw = Mock(return_value=mass_choice)
    helpers.update(
        random=SimpleNamespace(randint=draw), st_mass=330000.0,
        logD_array=np.array([observed]),
    )
    first_mass = 10.0
    gamma = 0.5
    second_mass = first_mass / gamma if mass_choice == 0 else first_mass * gamma
    mass_ratio_log = np.log10((first_mass + second_mass) / helpers['st_mass'])
    separation = 10.0 ** helpers['log_K'](mass_ratio_log, observed)

    actual = helpers['log_likelihood']([first_mass, gamma], helpers['logD_array'], return_directions=return_directions)
    if return_directions:
        actual, directions = actual
        assert directions == (mass_choice,)

    expected = -np.inf if separation > 40 else np.log10(expit(separation - np.sqrt(12.0)))
    assert np.asarray(actual).item() == pytest.approx(expected, rel=1e-13)
    draw.assert_called_once_with(0, 1)


@pytest.mark.parametrize('observed', [400.0, -400.0, np.inf, -np.inf])
@pytest.mark.parametrize('mass_choice', [0, 1])
@pytest.mark.parametrize('return_directions', [False, True])
def test_planet_mass_likelihood_preserves_extreme_limits_and_cutoff(observed, mass_choice, return_directions):
    import pairwise_model

    helpers = numerical_helpers('MCMC_planet_masses.py')
    assert helpers['orbital_separation'] is pairwise_model.orbital_separation
    assert helpers['log_K'] is pairwise_model.log_K
    draw = Mock(return_value=mass_choice)
    helpers.update(random=SimpleNamespace(randint=draw), st_mass=330000.0, logD_array=np.array([observed]))

    with np.errstate(all='raise'):
        actual = helpers['log_likelihood']([10.0, 0.5], helpers['logD_array'], return_directions=return_directions)

    if return_directions:
        actual, directions = actual
        assert directions == (mass_choice,)
    expected = -np.inf if observed > 0 else log_expit(-np.sqrt(12.0)) / np.log(10.0)
    assert np.asarray(actual).item() == expected
    draw.assert_called_once_with(0, 1)


@pytest.mark.parametrize('pair_count', [1, 3])
@pytest.mark.parametrize('sample_count', [1, 5])
@pytest.mark.parametrize('direction', [0, 1])
def test_neighbor_mass_recursion_preserves_each_sample_and_direction(pair_count, sample_count, direction):
    from pairwise_model import reconstruct_neighbor_masses

    first_mass = np.linspace(10.0, 20.0, sample_count)  # [M_earth]
    ratios = np.linspace(0.2, 0.8, pair_count * sample_count).reshape(pair_count, sample_count)
    directions = np.full(ratios.shape, direction)
    expected = [first_mass]
    for ratio in ratios:
        expected.append(expected[-1] / ratio if direction == 0 else expected[-1] * ratio)

    with np.errstate(all='raise'):
        actual = reconstruct_neighbor_masses(first_mass, ratios, directions)

    np.testing.assert_array_equal(actual, np.stack(expected))
    for index in range(sample_count):
        scalar = reconstruct_neighbor_masses(first_mass[index], ratios[:, index], directions[:, index])
        np.testing.assert_array_equal(scalar, actual[:, index])


@pytest.mark.parametrize('pair_count', [1, 2])
@pytest.mark.parametrize('return_directions', [False, True])
def test_planet_mass_likelihood_uses_supplied_observations_instead_of_global_catalog(pair_count, return_directions):
    helpers = numerical_helpers('MCMC_planet_masses.py')
    draw = Mock(return_value=0)
    helpers.update(random=SimpleNamespace(randint=draw), st_mass=330000.0, logD_array=np.array([2.0]))
    observations = np.linspace(-1.0, -0.5, pair_count)
    masses = 10.0 * 2.0 ** np.arange(pair_count + 1)  # [M_earth]
    log_mass_ratios = np.log10((masses[:-1] + masses[1:]) / helpers['st_mass'])
    separation = 10.0 ** (observations - (1 / 3) * log_mass_ratios + (1 / 3) * np.log10(3.0))
    expected = np.sum(log_expit(separation - np.sqrt(12)) / np.log(10.0))

    with np.errstate(all='raise'):
        actual = helpers['log_likelihood']([10.0] + [0.5] * pair_count, observations, return_directions=return_directions)

    if return_directions:
        actual, directions = actual
        assert directions == (0,) * pair_count
    assert np.isfinite(np.asarray(actual)).all()
    assert np.asarray(actual).item() == pytest.approx(expected, rel=1e-13)
    assert draw.call_count == pair_count


@pytest.mark.parametrize('pair_count', [1, 3])
@pytest.mark.parametrize('sample_count', [2, 7])
@pytest.mark.parametrize('reverse_samples', [False, True])
def test_planet_mass_kappa_corner_markers_summarize_every_sample(pair_count, sample_count, reverse_samples):
    path = Path(__file__).with_name('MCMC_planet_masses.py')
    print(f'Reading from {path}...')
    tree = ast.parse(path.read_text())
    loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For) and any(isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'logK' for target in statement.targets) for statement in node.body))
    observed = np.linspace(-2.0, -1.0, pair_count)
    log_mass_ratios = np.linspace(-5.0, -3.0, pair_count * sample_count).reshape(pair_count, sample_count)
    if reverse_samples:
        log_mass_ratios = log_mass_ratios[:, ::-1]
    expected = 10 ** (observed[:, None] - (1 / 3) * log_mass_ratios + (1 / 3) * np.log10(3.0))
    namespace = {'np': np, 'log_K': numerical_helpers(path.name)['log_K'], 'logD_array': observed, 'log_mu_tilde': log_mass_ratios, 'flat_samples_corner': np.zeros((1, sample_count)), 'labels': [], 'truths': []}

    with np.errstate(all='raise'):
        exec(compile(ast.Module(body=[loop], type_ignores=[]), str(path), 'exec'), namespace)

    assert namespace['flat_samples_corner'].shape == (pair_count + 1, sample_count)
    np.testing.assert_array_equal(namespace['flat_samples_corner'][1:], expected)
    np.testing.assert_array_equal(namespace['truths'], np.mean(expected, axis=1))
    assert len(namespace['labels']) == pair_count


@pytest.mark.parametrize('return_directions', [False, True])
@pytest.mark.parametrize(('theta', 'observations'), [
    ([10.0, 0.5, 0.5], []),
    ([10.0, 0.5, 0.5], [-1.0]),
    ([10.0, 0.5, 0.5], [-1.0, -0.5, -0.2]),
    ([10.0, 0.5, 0.5], [[-1.0], [-0.5]]),
    ([10.0, 0.5, 0.5], [[-1.0, -0.5], [-0.2, -0.1]]),
    ([10.0, 0.5, 0.5], -1.0),
    ([10.0, 0.5, 0.5], [np.nan, -1.0]),
    ([10.0, 0.5, 0.5], np.ma.array([-1.0, -0.5], mask=[True, False])),
    ([10.0, 0.5, 0.5], np.ma.masked_all(2)),
    ([10.0, 0.5, 0.5], np.ma.masked),
    ([10.0], []),
])
def test_planet_mass_likelihood_rejects_unaligned_observations_before_random_draws(theta, observations, return_directions):
    helpers = numerical_helpers('MCMC_planet_masses.py')
    draw = Mock(side_effect=AssertionError('Invalid observations consumed random state'))
    reconstruct = Mock(side_effect=AssertionError('Invalid observations reached mass reconstruction'))
    helpers.update(random=SimpleNamespace(randint=draw), reconstruct_neighbor_masses=reconstruct)

    with np.errstate(all='raise'), pytest.raises(ValueError, match='Observed separations'):
        helpers['log_likelihood'](theta, observations, return_directions=return_directions)

    draw.assert_not_called()
    reconstruct.assert_not_called()


@pytest.mark.parametrize('container', ['list', 'tuple', 'array', 'series', 'unmasked_array'])
@pytest.mark.parametrize('pair_count', [1, 2])
@pytest.mark.parametrize('return_directions', [False, True])
def test_planet_mass_likelihood_preserves_aligned_observation_containers(container, pair_count, return_directions):
    import pandas as pd

    helpers = numerical_helpers('MCMC_planet_masses.py')
    draw = Mock(return_value=0)
    helpers.update(random=SimpleNamespace(randint=draw), st_mass=330000.0)
    observations = np.linspace(-1.0, -0.5, pair_count)
    convert = {'list': list, 'tuple': tuple, 'array': np.asarray, 'series': pd.Series, 'unmasked_array': np.ma.asarray}[container]
    masses = 10.0 * 2.0 ** np.arange(pair_count + 1)  # [M_earth]
    log_mass_ratios = np.log10((masses[:-1] + masses[1:]) / helpers['st_mass'])
    separation = 10.0 ** (observations - (1 / 3) * log_mass_ratios + (1 / 3) * np.log10(3.0))
    expected = np.sum(log_expit(separation - np.sqrt(12)) / np.log(10.0))

    with np.errstate(all='raise'):
        actual = helpers['log_likelihood']([10.0] + [0.5] * pair_count, convert(observations), return_directions=return_directions)

    if return_directions:
        actual, directions = actual
        assert directions == (0,) * pair_count
    assert np.asarray(actual).item() == pytest.approx(expected, rel=1e-13)
    assert draw.call_count == pair_count


@pytest.mark.parametrize('planet_count', [2, 4])
@pytest.mark.parametrize('sample_count', [2, 7])
def test_planet_mass_ratio_corner_columns_retain_all_adjacent_pairs(planet_count, sample_count):
    import pairwise_model

    path = Path(__file__).with_name('MCMC_planet_masses.py')
    print(f'Reading from {path}...')
    tree = ast.parse(path.read_text())
    assignment = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'mu_tilde_distribution'
                for target in node.targets)
    )
    summary_loop = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Tuple)
        and any(isinstance(target, ast.Name) and target.id == 'log_mutilde'
                for target in node.target.elts)
    )
    masses = np.arange(1.0, planet_count * sample_count + 1.0).reshape(planet_count, sample_count)  # [M_earth]
    stellar_mass = 100000.0  # [M_earth]
    expected = (masses[:-1] + masses[1:]) / stellar_mass
    namespace = {
        'np': np,
        'masses': masses,
        'st_mass': stellar_mass,
        'adjacent_pair_mass_ratios': pairwise_model.adjacent_pair_mass_ratios,
        'labels': [],
        'truths': [],
    }

    with np.errstate(all='raise'):
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(path), 'exec'), namespace)

    np.testing.assert_array_equal(namespace['mu_tilde_distribution'], expected)
    namespace['log_mu_tilde'] = np.log10(namespace['mu_tilde_distribution'])
    with np.errstate(all='raise'):
        exec(compile(ast.Module(body=[summary_loop], type_ignores=[]), str(path), 'exec'), namespace)
    np.testing.assert_array_equal(namespace['truths'], np.mean(np.log10(expected), axis=1))
    assert len(namespace['labels']) == planet_count - 1


@pytest.mark.parametrize('return_directions', [False, True])
@pytest.mark.parametrize('theta', [
    (-1.0, 0.5), (0.0, 0.5), (0.999, 0.5), (317.80001, 0.5),
    (np.nan, 0.5), (np.inf, 0.5), (-np.inf, 0.5),
    (10.0, np.nan), (10.0, np.inf), (10.0, -np.inf),
    (10.0, 0.0), (10.0, 1.0), (10.0, -0.2), (10.0, 1.2),
    (10.0, 0.5, np.nan), (10.0, 0.5, -np.inf),
    (10.0, 0.5, 0.0), (10.0, 0.5, 1.0),
])
def test_planet_mass_prior_rejects_invalid_proposals_before_unsafe_arithmetic(theta, return_directions):
    helpers = numerical_helpers('MCMC_planet_masses.py')
    likelihood = Mock(side_effect=AssertionError('Invalid proposal reached likelihood'))
    helpers.update(order=2, lower_limit=1.0, gamma_mean=1.0, gamma_std=0.3, logD_array=np.zeros(len(theta) - 1), log_likelihood=likelihood)

    with np.errstate(all='raise'):
        assert helpers['log_prior'](theta) == -np.inf
        result = helpers['log_posterior'](theta, return_directions=return_directions)

    assert result == ((-np.inf, (0,) * (len(theta) - 1)) if return_directions else -np.inf)
    likelihood.assert_not_called()


@pytest.mark.parametrize('first_mass', [1.0, 10.0, 317.8])
@pytest.mark.parametrize('gammas', [(0.01,), (0.5,), (0.99,), (0.2, 0.8)])
def test_planet_mass_prior_preserves_written_density_and_inclusive_mass_bounds(first_mass, gammas):
    helpers = numerical_helpers('MCMC_planet_masses.py')
    helpers.update(order=2, lower_limit=1.0, gamma_mean=1.0, gamma_std=0.3)
    expected = np.log10(2 - 1) - np.log10(1.0) - 2 * np.log10(first_mass) + 2 * np.log10(1.0)
    for gamma in gammas:
        expected += stats.norm.logpdf(gamma, 1.0, 0.3)

    with np.errstate(all='raise'):
        actual = helpers['log_prior']((first_mass, *gammas))

    assert np.isfinite(actual)
    assert actual == expected


@pytest.mark.parametrize('pair_count', [1, 2])
@pytest.mark.parametrize('seed', [42, 123])
def test_planet_mass_reconstruction_uses_accepted_emcee_direction_metadata(pair_count, seed):
    emcee = pytest.importorskip('emcee')
    helpers = numerical_helpers('MCMC_planet_masses.py')
    helpers.update(random=random.Random(seed), st_mass=330000.0, logD_array=np.full(pair_count, -1.0), order=2, lower_limit=1.0, gamma_mean=1.0, gamma_std=0.3)
    path = Path(__file__).with_name('MCMC_planet_masses.py')
    print(f'Reading from {path}...')
    tree = ast.parse(path.read_text())
    constructor = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'sampler' for target in node.targets))
    outer = next(node for node in tree.body if isinstance(node, ast.While))
    branch = next(node for node in outer.body if isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Call) and isinstance(node.test.left.func, ast.Name) and node.test.left.func.id == 'len')
    body = branch.orelse
    start = next(index for index, node in enumerate(body) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'm0' for target in node.targets) and isinstance(node.value, ast.Subscript) and isinstance(node.value.value, ast.Name) and node.value.value.id == 'flat_samples')
    end = next(index for index in range(start, len(body)) if isinstance(body[index], ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'flat_samples_corner' for target in body[index].targets))
    callback = Mock(wraps=helpers['log_posterior'])
    helpers.update(emcee=emcee, nwalkers=8, ndim=pair_count + 1, num_steps=16, log_posterior=callback)
    exec(compile(ast.Module(body=[constructor], type_ignores=[]), str(path), 'exec'), helpers)
    sampler = helpers['sampler']
    sampler.random_state = np.random.RandomState(seed).get_state()
    rng = np.random.default_rng(seed)
    positions = np.column_stack((rng.uniform(10.0, 10.1, 8), rng.uniform(0.3, 0.7, (8, pair_count))))

    state = sampler.run_mcmc(positions, 2, progress=False)
    sampler.reset()
    sampler.run_mcmc(state, 16, progress=False)
    helpers['flat_samples'] = sampler.get_chain(flat=True).T
    with np.errstate(all='raise'):
        exec(compile(ast.Module(body=body[start:end], type_ignores=[]), str(path), 'exec'), helpers)

    draws = sampler.get_chain(flat=True)
    flags = sampler.get_blobs(flat=True)['mass_directions']
    assert callback.call_count > len(draws)
    assert flags.shape == (len(draws), pair_count)
    assert np.isin(flags, [0, 1]).all()
    masses = helpers['masses']
    assert masses.shape == (pair_count + 1, len(draws))
    np.testing.assert_array_equal(masses[0], draws[:, 0])
    for index in range(pair_count):
        expected = np.where(flags[:, index] == 0, masses[index] / draws[:, index + 1], masses[index] * draws[:, index + 1])
        np.testing.assert_array_equal(masses[index + 1], expected)
    log_mass_ratio = np.log10((masses[:-1] + masses[1:]) / helpers['st_mass'])
    separation = 10 ** (helpers['logD_array'][:, None] - (1 / 3) * log_mass_ratio + (1 / 3) * np.log10(3.0))
    expected_likelihood = np.sum(log_expit(separation - np.sqrt(12)) / np.log(10.0), axis=0)
    expected_prior = np.array([helpers['log_prior'](draw) for draw in draws])
    np.testing.assert_allclose(sampler.get_log_prob(flat=True), expected_prior + expected_likelihood, rtol=1e-14)
    chain = sampler.get_chain()
    repeated = np.all(chain[1:] == chain[:-1], axis=2)
    assert repeated.any()
    blobs = sampler.get_blobs()['mass_directions']
    np.testing.assert_array_equal(blobs[1:][repeated], blobs[:-1][repeated])


def test_neighbor_mass_recursion_does_not_evaluate_unused_extreme_branches():
    from pairwise_model import reconstruct_neighbor_masses

    first_mass = np.array([1e200, 1e-200])  # [M_earth]
    ratios = np.array([[1e-200, 1e-200]])
    directions = np.array([[1, 0]])
    with np.errstate(all='raise'):
        actual = reconstruct_neighbor_masses(first_mass, ratios, directions)
        scalar = reconstruct_neighbor_masses(first_mass[0], ratios[:, 0], directions[:, 0])

    np.testing.assert_array_equal(actual[-1], [1.0, 1.0])
    np.testing.assert_array_equal(scalar, actual[:, 0])


@pytest.mark.parametrize(('ratios', 'directions'), [
    ([[0.5, 0.5]], [0]),
    ([0.5, 0.5], [0, 1]),
    ([[0.5, 0.5]], [[0]]),
])
def test_neighbor_mass_recursion_rejects_unaligned_sample_metadata(ratios, directions):
    from pairwise_model import reconstruct_neighbor_masses

    with pytest.raises(ValueError, match='align'):
        reconstruct_neighbor_masses([10.0, 20.0], ratios, directions)


@pytest.mark.parametrize('directions', [
    [np.nan], [np.inf], [-np.inf], [0.5], [2], [-1], ['0'],
    np.array([0.0 + 0.0j]), np.array([pd.NA], dtype=object),
    np.ma.array([0], mask=[True]), np.ma.masked_all(1),
])
def test_neighbor_mass_recursion_rejects_missing_or_nonbinary_choices_before_arithmetic(directions):
    from pairwise_model import reconstruct_neighbor_masses

    with np.errstate(all='raise'), pytest.raises(ValueError, match='Mass directions'):
        reconstruct_neighbor_masses(10.0, [0.0], directions)


@pytest.mark.parametrize('dtype', [bool, np.int8, np.int64, np.float64])
@pytest.mark.parametrize('unmasked', [False, True])
def test_neighbor_mass_recursion_preserves_binary_choice_types(dtype, unmasked):
    from pairwise_model import reconstruct_neighbor_masses

    choices = np.array([[0, 1], [1, 0]], dtype=dtype)
    if unmasked:
        choices = np.ma.asarray(choices)
    with np.errstate(all='raise'):
        actual = reconstruct_neighbor_masses([10.0, 20.0], [[0.5, 0.5], [0.25, 0.25]], choices)

    np.testing.assert_array_equal(actual, [[10.0, 20.0], [20.0, 10.0], [5.0, 40.0]])


@pytest.mark.parametrize('field', ['first_mass', 'mass_ratios'])
@pytest.mark.parametrize('kind', ['partial', 'full', 'scalar'])
def test_neighbor_mass_recursion_rejects_masked_physical_inputs(field, kind):
    from pairwise_model import reconstruct_neighbor_masses

    arguments = {
        'first_mass': np.array([10.0, 20.0]),  # [M_earth]
        'mass_ratios': np.array([[0.5, 0.5]]),
        'directions': np.array([[0, 1]]),
    }
    if kind == 'scalar':
        arguments[field] = np.ma.masked
    else:
        value = arguments[field]
        mask = np.ones(value.shape, dtype=bool) if kind == 'full' else np.array([True, False]).reshape(value.shape)
        arguments[field] = np.ma.array(value, mask=mask)

    with np.errstate(all='raise'), pytest.raises(ValueError, match='Mass inputs and ratios.*masked'):
        reconstruct_neighbor_masses(**arguments)


@pytest.mark.parametrize('unmasked_field', ['first_mass', 'mass_ratios', 'both'])
@pytest.mark.parametrize('scalar', [False, True])
def test_neighbor_mass_recursion_preserves_unmasked_physical_containers(unmasked_field, scalar):
    from pairwise_model import reconstruct_neighbor_masses

    first_mass = np.array(10.0) if scalar else np.array([10.0, 20.0])  # [M_earth]
    ratios = np.array([0.5, 0.25]) if scalar else np.array([[0.5, 0.5], [0.25, 0.25]])
    directions = np.array([0, 1]) if scalar else np.array([[0, 1], [1, 0]])
    expected = reconstruct_neighbor_masses(first_mass, ratios, directions)
    if unmasked_field in ('first_mass', 'both'):
        first_mass = np.ma.asarray(first_mass)
    if unmasked_field in ('mass_ratios', 'both'):
        ratios = np.ma.asarray(ratios)

    with np.errstate(all='raise'):
        actual = reconstruct_neighbor_masses(first_mass, ratios, directions)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('field', ['first_mass', 'mass_ratios'])
@pytest.mark.parametrize('value', [-1.0, 0.0, np.nan, np.inf, -np.inf, True, '1.0', 1.0 + 0.0j, None])
@pytest.mark.parametrize('scalar', [False, True])
def test_neighbor_mass_recursion_rejects_invalid_numeric_inputs(field, value, scalar):
    from pairwise_model import reconstruct_neighbor_masses

    arguments = {
        'first_mass': 10.0 if scalar else np.array([10.0, 20.0]),  # [M_earth]
        'mass_ratios': np.array([0.5]) if scalar else np.array([[0.5, 0.5]]),
        'directions': np.array([1]) if scalar else np.array([[1, 0]]),
    }
    shape = np.shape(arguments[field])
    arguments[field] = np.full(shape, value)
    message = 'Initial mass' if field == 'first_mass' else 'Mass ratios'

    with np.errstate(all='raise'), pytest.raises(ValueError, match=message):
        reconstruct_neighbor_masses(**arguments)


@pytest.mark.parametrize('ratio', [0.5, 1.0, 2.0])
@pytest.mark.parametrize('direction', [0, 1])
def test_neighbor_mass_recursion_preserves_positive_ratio_arithmetic_without_new_prior_bounds(ratio, direction):
    from pairwise_model import reconstruct_neighbor_masses

    with np.errstate(all='raise'):
        actual = reconstruct_neighbor_masses(10.0, [ratio], [direction])

    expected = 10.0 / ratio if direction == 0 else 10.0 * ratio  # [M_earth]
    np.testing.assert_array_equal(actual, [10.0, expected])


def test_planet_mass_sampler_uses_the_maintained_tdpy_function():
    from tdpy import samp_powr

    helpers = numerical_helpers('MCMC_planet_masses.py')
    assert helpers['samp_powr'] is samp_powr
    samples = helpers['samp_powr'](32, 1.0, 317.8, 2.0)
    assert samples.shape == (32,)
    assert np.isfinite(samples).all()
    assert np.all((samples >= 1.0) & (samples <= 317.8))


@pytest.mark.parametrize('filename', SCRIPT_FILES[:2])
@pytest.mark.parametrize('theta', [(-4.0, 0.5), (-5.0, 0.2), (-3.0, 0.9)])
def test_pairwise_prior_preserves_the_written_density_factors_in_log_space(filename, theta):
    log_mass_ratio, gamma = theta
    mass_density = np.exp(-0.5 * ((log_mass_ratio + 4.31921754762478) / 0.28778378882844535) ** 2)
    mass_density /= 10.0 ** log_mass_ratio * 0.28778378882844535 * np.sqrt(2.0 * np.pi)
    gamma_density = np.exp(-0.5 * ((np.log10(gamma) - 0.5) / 0.3) ** 2)
    gamma_density /= gamma * 0.3 * np.sqrt(2.0 * np.pi)

    actual = numerical_helpers(filename)['log_prior'](theta)

    assert np.isfinite(actual)
    assert actual == pytest.approx(np.log(mass_density) + np.log(gamma_density), rel=1e-13)


@pytest.mark.parametrize('filename', SCRIPT_FILES[:2])
@pytest.mark.parametrize('theta', [(-4.0, 0.0), (-4.0, 1.0), (-4.0, -0.5), (-4.0, np.nan), (np.nan, 0.5), (np.inf, 0.5)])
def test_pairwise_prior_rejects_invalid_mass_ratios_and_nonfinite_values(filename, theta):
    assert numerical_helpers(filename)['log_prior'](theta) == -np.inf


@pytest.mark.parametrize('filename', SCRIPT_FILES[:2])
@pytest.mark.parametrize('mass_coordinate', [-5e153, -4e153, 4e153, 5e153])
def test_pairwise_prior_retains_representable_extreme_gaussian_tails(filename, mass_coordinate):
    prior = numerical_helpers(filename)['log_prior']
    with localcontext() as context:
        context.prec = 90
        coordinate = Decimal.from_float(mass_coordinate)
        mean = Decimal.from_float(-4.31921754762478)
        standard_deviation = Decimal.from_float(0.28778378882844535)
        root_two_pi = (2 * Decimal.from_float(np.pi)).sqrt()
        mass_gaussian = -((coordinate - mean) / standard_deviation) ** 2 / 2 - (standard_deviation * root_two_pi).ln()
        gamma = Decimal('0.5')
        gamma_gaussian = -((gamma.log10() - Decimal('0.5')) / Decimal.from_float(0.3)) ** 2 / 2 - (Decimal.from_float(0.3) * root_two_pi).ln()
        expected = float(mass_gaussian - Decimal(10).ln() * coordinate + gamma_gaussian - gamma.ln())

    with np.errstate(all='raise'):
        actual = prior((mass_coordinate, 0.5))

    assert np.isfinite(actual)
    assert actual == pytest.approx(expected, rel=2e-15)


@pytest.mark.parametrize('filename', SCRIPT_FILES[:2])
@pytest.mark.parametrize('mass_coordinate', [-np.finfo(float).max, -1e200, -1e154, 1e154, 1e200, np.finfo(float).max])
def test_pairwise_prior_returns_negative_infinity_without_invalid_tail_arithmetic(filename, mass_coordinate):
    helpers = numerical_helpers(filename)
    likelihood = Mock(side_effect=AssertionError('Unrepresentable prior tail reached likelihood'))
    helpers['log_likelihood'] = likelihood
    with np.errstate(all='raise'):
        assert helpers['log_prior']((mass_coordinate, 0.5)) == -np.inf
        assert helpers['log_posterior']((mass_coordinate, 0.5)) == -np.inf
    likelihood.assert_not_called()


@pytest.mark.parametrize('filename', SCRIPT_FILES[:2])
def test_pairwise_prior_preserves_ordinary_evaluation_exactly(filename):
    prior = numerical_helpers(filename)['log_prior']
    for mass_coordinate in np.linspace(-100.0, 100.0, 11):
        for gamma in [0.01, 0.2, 0.5, 0.9, 0.999]:
            mass_term = stats.norm.logpdf(mass_coordinate, -4.31921754762478, 0.28778378882844535) - np.log(10.0) * mass_coordinate
            gamma_term = stats.norm.logpdf(np.log10(gamma), 0.5, 0.3) - np.log(gamma)
            assert prior((mass_coordinate, gamma)) == mass_term + gamma_term


@pytest.mark.parametrize('filename', SCRIPT_FILES[:2])
def test_pairwise_posterior_and_initialization_use_the_same_parameter_order(filename):
    import pairwise_model

    helpers = numerical_helpers(filename)
    path = Path(__file__).with_name(filename)
    print(f'Reading from {path}...')
    tree = ast.parse(path.read_text())
    position = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'pos' for target in node.targets))
    helpers.update(
        nwalkers=2,
        initial_pairwise_positions=pairwise_model.initial_pairwise_positions,
    )
    np.random.seed(23)
    exec(compile(ast.Module(body=[position], type_ignores=[]), str(path), 'exec'), helpers)
    np.random.seed(23)
    expected_positions = pairwise_model.initial_pairwise_positions(
        2, -4.31921754762478, 0.28778378882844535
    )
    np.testing.assert_array_equal(helpers['pos'], expected_positions)

    for theta in helpers['pos']:
        prior = helpers['log_prior'](theta)
        likelihood = helpers['log_likelihood'](theta)
        assert np.isfinite(prior)
        assert np.ndim(likelihood) == 0
        assert helpers['log_posterior'](theta) == pytest.approx(prior + likelihood)


def test_all_pairs_likelihood_sums_independent_observations_from_a_dataframe():
    import pandas as pd

    helpers = numerical_helpers('MCMC_all_pairs.py')
    observations = pd.DataFrame({'0': [-2.0, -1.5, -1.0]})
    expected = sum(helpers['log_likelihood']((-4.0, 0.5), value) for value in observations['0'])
    assert helpers['log_likelihood']((-4.0, 0.5), observations) == pytest.approx(expected)


@pytest.mark.parametrize('as_dataframe', [False, True])
@pytest.mark.parametrize(('separation_power', 'count'), [(306.0, 2), (307.5, 4), (307.5, 8), (308.0, 1), (308.0, 2), (308.0, 8)])
def test_all_pairs_likelihood_retains_finite_and_unrepresentable_total_limits(as_dataframe, separation_power, count):
    import pandas as pd

    helpers = numerical_helpers('MCMC_all_pairs.py')
    mass_coordinate = -4.0
    observed = np.full(count, separation_power + mass_coordinate / 3 - np.log10(3.0) / 3)
    if as_dataframe:
        observed = pd.DataFrame({'0': observed})
    with np.errstate(all='raise'):
        values = helpers['orbital_log_likelihood'](mass_coordinate, observed)
        actual = helpers['log_likelihood']((mass_coordinate, 0.5), observed)
    assert np.isfinite(np.asarray(values)).all()
    with localcontext() as context:
        context.prec = 90
        expected = float(sum(Decimal.from_float(float(value)) for value in np.asarray(values).ravel()))
    assert isinstance(actual, float)
    if np.isfinite(expected):
        assert actual == pytest.approx(expected, rel=2e-15)
    else:
        assert actual == -np.inf


def test_single_pair_catalog_value_uses_positional_row_and_column_selection():
    import pandas as pd

    path = Path(__file__).with_name('MCMC_single_pair.py')
    print(f'Reading from {path}...')
    tree = ast.parse(path.read_text())
    selection = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'logD_value' for target in node.targets))
    namespace = {'datafile': pd.DataFrame({'0': [-0.3, -0.4]})}
    exec(compile(ast.Module(body=[selection], type_ignores=[]), str(path), 'exec'), namespace)
    assert namespace['logD_value'] == -0.3


@pytest.mark.parametrize('filename', SCRIPT_FILES[:2])
def test_pairwise_sampler_accepts_and_moves_with_the_corrected_posterior(filename):
    emcee = pytest.importorskip('emcee')
    helpers = numerical_helpers(filename)
    random = np.random.default_rng(123)
    positions = np.column_stack((random.normal(-4.0, 0.01, 8), random.uniform(0.3, 0.7, 8)))
    sampler = emcee.EnsembleSampler(8, 2, helpers['log_posterior'])
    sampler.random_state = np.random.RandomState(123).get_state()

    sampler.run_mcmc(positions, 32, progress=False)

    assert sampler.get_chain().shape == (32, 8, 2)
    assert np.isfinite(sampler.get_log_prob()).all()
    assert np.any(sampler.acceptance_fraction > 0.0)
    assert not np.array_equal(sampler.get_last_sample().coords, positions)


@pytest.mark.parametrize('filename', SCRIPT_FILES[:2])
@pytest.mark.parametrize('name', ['log_K', 'log_prior'])
def test_pairwise_scripts_reexport_the_same_common_model(filename, name):
    import pairwise_model

    assert numerical_helpers(filename)[name] is getattr(pairwise_model, name)


@pytest.mark.parametrize('filename', SCRIPT_FILES[:2])
@pytest.mark.parametrize('prior_value', [-np.inf, np.inf, np.nan])
def test_pairwise_posterior_does_not_evaluate_likelihood_outside_prior_support(filename, prior_value):
    helpers = numerical_helpers(filename)
    prior = Mock(return_value=prior_value)
    likelihood = Mock(side_effect=AssertionError('Likelihood evaluated outside support'))
    helpers.update(log_prior=prior, log_likelihood=likelihood)
    theta = (-4.0, 0.5)

    assert helpers['log_posterior'](theta) == -np.inf
    prior.assert_called_once_with(theta)
    likelihood.assert_not_called()


@pytest.mark.parametrize('filename', SCRIPT_FILES[:2])
def test_pairwise_posterior_combiner_uses_each_scripts_current_callbacks(filename):
    helpers = numerical_helpers(filename)
    prior = Mock(return_value=-2.0)
    likelihood = Mock(return_value=-3.0)
    helpers.update(log_prior=prior, log_likelihood=likelihood)
    theta = (-4.0, 0.5)

    assert helpers['log_posterior'](theta) == -5.0
    prior.assert_called_once_with(theta)
    likelihood.assert_called_once_with(theta)


@pytest.mark.parametrize('filename', SCRIPT_FILES[:2])
@pytest.mark.parametrize('prior_type', [float, np.float64])
@pytest.mark.parametrize('likelihood_type', [float, np.float64])
@pytest.mark.parametrize(('prior_value', 'likelihood_value'), [(-2.0, -3.0), (-3e307, -4e307), (-1e308, -1e308)])
def test_pairwise_posterior_preserves_finite_and_overflowed_scalar_totals(filename, prior_type, likelihood_type, prior_value, likelihood_value):
    helpers = numerical_helpers(filename)
    prior = Mock(return_value=prior_type(prior_value))
    likelihood = Mock(return_value=likelihood_type(likelihood_value))
    helpers.update(log_prior=prior, log_likelihood=likelihood)
    theta = (-4.0, 0.5)
    with localcontext() as context:
        context.prec = 400
        expected = float(Decimal.from_float(prior_value) + Decimal.from_float(likelihood_value))

    with np.errstate(all='raise'):
        policy = np.geterr()
        actual = helpers['log_posterior'](theta)
        assert np.geterr() == policy

    assert actual == expected
    assert np.ndim(actual) == 0
    prior.assert_called_once_with(theta)
    likelihood.assert_called_once_with(theta)


@pytest.mark.parametrize('filename', SCRIPT_FILES[:2])
def test_pairwise_posterior_does_not_suppress_overflow_inside_likelihood_callback(filename):
    helpers = numerical_helpers(filename)
    prior = Mock(return_value=np.float64(-2.0))

    def likelihood(theta):
        return np.multiply(np.float64(1e308), np.float64(2.0))

    helpers.update(log_prior=prior, log_likelihood=likelihood)
    with np.errstate(all='raise'), pytest.raises(FloatingPointError, match='overflow'):
        helpers['log_posterior']((-4.0, 0.5))
    prior.assert_called_once_with((-4.0, 0.5))


@pytest.mark.parametrize('theta', [
    (np.nan, -4.0), (np.inf, -4.0), (-np.inf, -4.0),
    (0.0, -4.0), (1.0, -4.0), (-0.5, -4.0), (np.finfo(float).max, -4.0),
    (0.5, np.nan), (0.5, np.inf), (0.5, -np.inf),
])
def test_multi_system_prior_rejects_invalid_support_before_gaussian_evaluation(theta):
    helpers = numerical_helpers('MCMC_single_system_all_pairs.py')
    likelihood = Mock(side_effect=AssertionError('Invalid multi-system proposal reached likelihood'))
    helpers['log_likelihood'] = likelihood

    with np.errstate(all='raise'):
        assert helpers['log_prior'](theta) == -np.inf
        assert helpers['log_posterior'](theta) == -np.inf

    likelihood.assert_not_called()


@pytest.mark.parametrize('mass_coordinate', [-5.0, -4.0, -3.0])
@pytest.mark.parametrize('gamma', [0.01, 0.5, 0.99])
def test_multi_system_prior_preserves_distinct_written_gaussians_exactly(mass_coordinate, gamma):
    helpers = numerical_helpers('MCMC_single_system_all_pairs.py')
    expected = stats.norm.logpdf(mass_coordinate, -4.3041661629482, 0.30148352295199915) + stats.norm.logpdf(gamma, 1, 0.3)

    with np.errstate(all='raise'):
        actual = helpers['log_prior']((gamma, mass_coordinate))

    assert actual == expected


def test_multi_system_posterior_uses_shared_prior_first_combiner():
    import pairwise_model

    helpers = numerical_helpers('MCMC_single_system_all_pairs.py')
    assert helpers['combine_log_posterior'] is pairwise_model.combine_log_posterior
    prior = Mock(return_value=-np.inf)
    likelihood = Mock(return_value=5.0)
    helpers.update(log_prior=prior, log_likelihood=likelihood)

    assert helpers['log_posterior']((0.5, -4.0)) == -np.inf
    likelihood.assert_not_called()

    prior.return_value = -3.0
    assert helpers['log_posterior']((0.5, -4.0)) == 2.0
    likelihood.assert_called_once_with((0.5, -4.0))


@pytest.mark.parametrize('mass_coordinate', [-5e153, 5e153, -1e200, 1e200, -np.finfo(float).max, np.finfo(float).max])
def test_multi_system_prior_uses_shared_gaussian_tail_limits(mass_coordinate):
    import pairwise_model

    helpers = numerical_helpers('MCMC_single_system_all_pairs.py')
    assert helpers['normal_log_density'] is pairwise_model.normal_log_density
    with localcontext() as context:
        context.prec = 90
        coordinate = Decimal.from_float(mass_coordinate)
        mean = Decimal.from_float(-4.3041661629482)
        deviation = Decimal.from_float(0.30148352295199915)
        expected = float(-((coordinate - mean) / deviation) ** 2 / 2)
    with np.errstate(all='raise'):
        actual = helpers['log_prior']((0.5, mass_coordinate))
    if np.isfinite(expected):
        assert actual == pytest.approx(expected, rel=2e-15)
    else:
        assert actual == -np.inf


@pytest.mark.parametrize(('value', 'mean', 'scale'), [
    (1e308, -1e308, 1e308), (-1e308, 1e308, 1e308),
    (np.finfo(float).max, -np.finfo(float).max, np.finfo(float).max),
    (1.0, 0.0, 1e308), (0.0, 0.0, 1e-300),
])
def test_shared_gaussian_retains_representable_wide_and_narrow_parameter_values(value, mean, scale):
    from pairwise_model import normal_log_density

    with localcontext() as context:
        context.prec = 90
        coordinate = Decimal.from_float(value)
        location = Decimal.from_float(mean)
        deviation = Decimal.from_float(scale)
        expected = float(-((coordinate - location) / deviation) ** 2 / 2 - deviation.ln() - (2 * Decimal.from_float(np.pi)).ln() / 2)

    with np.errstate(all='raise'):
        actual = normal_log_density(value, mean, scale)

    assert np.isfinite(actual)
    assert actual == pytest.approx(expected, rel=2e-15)


@pytest.mark.parametrize(('mean', 'scale'), [
    (np.nan, 1.0), (np.inf, 1.0), (-np.inf, 1.0),
    (0.0, 0.0), (0.0, -1.0), (0.0, np.nan), (0.0, np.inf), (0.0, -np.inf),
])
def test_shared_gaussian_rejects_invalid_distribution_parameters(mean, scale):
    from pairwise_model import normal_log_density

    with np.errstate(all='raise'), pytest.raises(ValueError, match='Gaussian location'):
        normal_log_density(0.0, mean, scale)


@pytest.mark.parametrize('coordinate', [-np.inf, np.inf])
def test_shared_gaussian_preserves_infinite_coordinate_tail_limits(coordinate):
    from pairwise_model import normal_log_density

    with np.errstate(all='raise'):
        assert normal_log_density(coordinate, 0.0, 1.0) == -np.inf


def test_common_pairwise_model_import_does_not_start_sampler_or_plotting():
    subprocess.run(
        [sys.executable, '-c', "import sys; import pairwise_model; assert all(name not in sys.modules for name in ('emcee', 'corner', 'matplotlib.pyplot', 'MCMC_all_pairs', 'MCMC_single_pair'))"],
        cwd=Path(__file__).parent, capture_output=True, text=True, check=True,
    )


@pytest.mark.parametrize('filename', SCRIPT_FILES[:3])
@pytest.mark.parametrize('observed', [400.0, -400.0, np.inf, -np.inf])
def test_pairwise_likelihood_retains_extreme_limits_without_exponent_warnings(filename, observed):
    helpers = numerical_helpers(filename)
    theta = (0.5, -4.0) if filename == 'MCMC_single_system_all_pairs.py' else (-4.0, 0.5)
    with np.errstate(all='raise'):
        actual = helpers['log_likelihood'](theta, observed)
    expected = -np.inf if observed > 0 else log_expit(np.sqrt(12.0))
    if filename == 'MCMC_single_system_all_pairs.py':
        expected /= np.log(10.0)
    assert actual == expected


@pytest.mark.parametrize('filename', SCRIPT_FILES[:3])
def test_shared_orbital_likelihood_preserves_vector_shape_and_all_pairs_aggregation(filename):
    import pairwise_model

    helpers = numerical_helpers(filename)
    assert helpers['log_K'] is pairwise_model.log_K
    assert helpers['orbital_log_likelihood'] is pairwise_model.orbital_log_likelihood
    observed = np.array([-400.0, -2.0, 0.0, 400.0])
    theta = (0.5, -4.0) if filename == 'MCMC_single_system_all_pairs.py' else (-4.0, 0.5)
    with np.errstate(all='raise'):
        values = helpers['orbital_log_likelihood'](-4.0, observed)
        actual = helpers['log_likelihood'](theta, observed)
    assert values.shape == observed.shape
    assert np.isfinite(values[:-1]).all()
    assert values[-1] == -np.inf
    if filename == 'MCMC_all_pairs.py':
        assert actual == -np.inf
        assert np.ndim(actual) == 0
    else:
        if filename == 'MCMC_single_system_all_pairs.py':
            values = values / np.log(10.0)
        np.testing.assert_array_equal(actual, values)
