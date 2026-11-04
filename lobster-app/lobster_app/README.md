# LOBSTER Research studio

A local Dash application for exploring extracted LOBSTER files and running reproducible PCA, forward response, and Patnaik Pearson dimension experiments.

## Launch

Double click **Launch LOBSTER.command** in the `lobster-app` folder. It opens a terminal and then your browser. Leave that terminal running while using the app. Press **Ctrl+C** in the terminal to stop it. Opening the launcher again recognizes an existing copy on the same port.

Alternatively, run this from the `lobster-app` folder.

```sh
lobster_app/.venv/bin/python -B lobster_app/app.py
```

Open http://127.0.0.1:8050. If another application uses that port, add `--port 8051`. The app can also be displayed in the Codex browser when its local server is running.

## Views

* **Overview** selects a file, time interval, and occupied depth. It shows quotes, spreads, imbalance, realised volatility, execution activity, depth, event counts, and source checks.
* **Order book** inspects exact original event rows. Use the slider, row number, previous and next buttons, or playback. Events per step controls replay speed. Each snapshot shows the order book after the selected message.
* **Experiments** runs PCA, forward midquote response against depth imbalance, or Patnaik Pearson dimension through time. Dimension analysis supports levels within a snapshot and recent histories of snapshots. Its inspector shows the singular spectrum and observation counts at any computed time. Settings are applied only when Run experiment is pressed.
* **Compare runs** compares completed experiments and downloads a JSON record of their results, settings, source hashes, software versions, and implementation fingerprints.

The most recent eight experiment runs stay in memory in the current browser tab. Refreshing or closing that tab loses this history. Download the records to retain them. They can be read in Python or any JSON viewer. This version does not import saved records into the interface.

## Data and performance

Place extracted message and matching orderbook CSV pairs inside `lobster-app/data`, optionally in dataset subfolders, then restart the application. Only that data folder is searched. Archives are not automatically extracted. The `lobster_exploration.py` command line tool also defaults to this data folder.

Two complete datasets and up to twelve experiment results are cached in memory. Source size and modification time changes invalidate cached datasets. Each loaded pair is also identified by full SHA256 hashes for experiment provenance. The loader holds complete datasets in memory, so this version is intended for individual sessions or files that fit available RAM. It is not an out of core data warehouse.

Charts deliberately bound display resolution. The quotes, spread and imbalance chart draws at most 3,500 evenly spaced event rows and may miss brief excursions. The realised volatility chart draws every point on its one second grid, preserving gaps. The depth heatmap samples at most 700 clock points. Tables and research calculations use the full selected data at the explicitly chosen analytical sampling frequency. Source checks refer to the complete loaded files.

Window boundaries are inclusive for event filtering. Overview time means cover the first through last selected event, without extending quotes to the slider boundaries. Known halt intervals are excluded. Cross and auction messages are excluded from execution totals. Historical prices are unadjusted dollars, and depth describes visible sizes at occupied price levels rather than fixed tick distances.

## Research definitions

Experiments sample the final original event at or before each regular clock point. Equal timestamps retain file order. Grids are anchored to midnight and bounded by loaded observations. Invalid best quotes, locked or crossed quotes, known halts, and snapshots older than the selected maximum age are excluded. Missing deeper levels are handled according to the chosen representation. The cleaned loader cannot distinguish a malformed missing zero size level from a documented dummy level.

### Realised volatility

The Overview chart uses the midquote sampled every second. **Volatility window** chooses 1, 5, 15 or 30 minutes, with 5 minutes selected initially. Let $m_t$ be the sampled midquote at integer second $t$. For a window of $W$ seconds, the plotted value is

$$
\mathrm{RV}_t=10^4\sqrt{\sum_{j=0}^{W-1}\left[\log\left(\frac{m_{t-j}}{m_{t-j-1}}\right)\right]^2}.
$$

Values are in basis points and are not annualised. Returns are not demeaned, and the sum is not divided by the number of returns. A larger window accumulates more squared returns, so values for different window lengths are not directly comparable as rates of volatility.

Sampling uses the final original message at or before each second, including the last message at tied timestamps. The full window and its initial midquote must lie within the selected period and observed coverage. Missing, locked, crossed or halted quotes, quotes more than 60 seconds old, and positive duration invalid or stale intervals between sample times invalidate returns. Any window containing an invalid return remains a gap until a complete valid window is available. No missing returns are replaced with zero. Unchanged valid quotes produce genuine zero returns.

The calculation uses only prices and is independent of the displayed depth and experiment settings. It retains every sampled return before plotting and uses no observations after the window endpoint. Short selections with no complete valid window show an explanatory message.

### PCA and forward response

These two methods use chronological training and evaluation partitions of the complete clock grid before invalid samples are removed. Their centering, scaling, constant feature detection, PCA directions, and response regression coefficients are fitted using training observations only. A missing deeper price is permitted only when its size is zero.

PCA uses bid and ask sizes at each selected occupied level. It supports raw shares or `log1p` shares and optional standardization. Training constant columns are removed and reported. The charts show training variance and later reconstruction error. PCA variance thresholds are not intrinsic dimension estimates.

Forward response is 10,000 times the logarithm of the ratio of the future midquote to the current midquote. The horizon must be an integer multiple of the sampling interval. A training target must end strictly before the split. Evaluation targets begin at or after the split. Targets crossing positive duration invalid quote or stale intervals are excluded, even when both endpoints are valid. Fixed equal width imbalance bins summarize means, medians, and counts. The line fit is ordinary least squares on training observations. Evaluation error is also compared against the training mean response baseline.

### Patnaik Pearson dimension

For a point matrix `X`, the application subtracts each column mean and computes its singular values `s`. It calculates `PP = sum(s)**2 / sum(s**2)`, following [Definition 3.3 and equation (21)](https://arxiv.org/html/2606.19268v4#S3.SS3). These are singular values, not covariance eigenvalues. Numerical rank is reported separately and does not truncate small singular values. The inspector divides the spectrum by its largest value, which leaves PP unchanged.

Snapshot settings now select individual feature groups rather than one of four encodings. Each group has **Off**, **Original**, **Standardised** and **Both** choices. Standardised applies causal historical standardisation to that group's columns. Both includes separate original and standardised columns. **Column standardisation** independently controls scaling within each PP estimate.

| Feature group | Original coordinates at L levels | Tom's setup |
| --- | --- | --- |
| Price differences | 4L − 2 | Standardised |
| Cumulative volumes | 2L | Standardised |
| Logarithms of cumulative volumes | 2L | Standardised |
| Volume imbalance | L | Both |
| Imbalance using logarithms of volumes | L | Both |
| Log volumes at each level | 2L | Off |
| Volumes as proportions of total depth | 2L | Off |
| Distances from the midquote | 2L | Off |

Tom's setup is the default and selects **12L − 2 coordinates**, or **118 at ten levels**. It uses every message, one snapshot per observation, a standardisation history of 501 observations and centring only within each estimate. Selecting no features produces an explanatory error. The group choices and unique column names are retained in exported results. The mode that compares price levels within one snapshot retains its separate, fixed construction described below.

The PP controls also include **Burn in time (seconds)** and **Burn out time (seconds)**, both initially zero. For a selected interval `[S, E]`, these retain only `[S + burn_in, E - burn_out]`, including both boundaries. Excluded observations are not used for sampling, historical standardisation or PP matrices. Historical standardisation therefore starts again within the retained interval and still applies its separate initialisation period. Settings that remove the entire interval are rejected. The durations and retained bounds are recorded in exports.

For each level number, the bid and ask provide the coordinates `[log1p(bid_size), log1p(ask_size), 10000*(mid-bid)/mid, 10000*(ask-mid)/mid]`. Snapshot mode removes levels with missing prices or invalid sizes and every observation containing such a level. In history mode, price distance embeddings require every selected price. Size embeddings permit missing prices only with zero size. Relative depth coordinates sum to one, imposing an affine constraint.

An observation can combine `K` sampled snapshots, with the current snapshot first and progressively earlier snapshots following. This multiplies the coordinate count by `K`. The history within one vector spans `(K - 1) × lag spacing`. The **estimation window** is separate and contains the collection of vectors used for one PP calculation. Every constituent clock time must remain inside that complete trailing window and the selected interval. Both modes exclude source observations before the selected start. No delay vector bridges a positive duration invalid or stale interval. Early endpoints without a complete estimation window remain undefined. No estimate uses observations after its endpoint. Sampling and evaluation grids remain fixed when observations are invalid.

**Original** retains the feature values before historical scaling. **Column standardisation** can divide centred coordinates by their sample standard deviations in that estimate. A feature group set to **Standardised** is transformed using strictly earlier sampled observations, as defined below, before assembling observations for PP. Constant coordinates remain zero and count towards the reported coordinate total. PP is descriptive and has no training split. Repeated observations are retained according to the selected sampling method.

### Collaborator representation and message sampling

In the PP experiment, **Use collaborator settings** selects the collaborator representation, every message, standardisation using past observations, a history of 501 observations and one snapshot per observation. It selects ten levels when available. It also selects an interval between PP estimates that fits the calculation limit for the current selection where possible. Both that interval and the estimation window remain adjustable. The repository constructs observations but does not specify a rolling PP estimation window.

The representation follows [PP_LOB/LOB/helper.py](https://github.com/RobertGBoyce/PP_LOB/blob/main/LOB/helper.py). For one snapshot, let $a_\ell$ and $b_\ell$ be the ask and bid prices in original LOBSTER integer units, and let $q^a_\ell$ and $q^b_\ell$ be their volumes in shares. Define cumulative volumes on each side by

$$
C^a_\ell=\sum_{k=1}^{\ell}q^a_k,\qquad C^b_\ell=\sum_{k=1}^{\ell}q^b_k.
$$

The vector contains the following groups in this order.

1. $a_\ell-a_1$ for levels 2 to $L$.
2. $a_\ell-b_1$ for levels 1 to $L$.
3. $b_\ell-b_1$ for levels 2 to $L$.
4. $b_\ell-a_1$ for levels 1 to $L$.
5. Cumulative ask volumes, then cumulative bid volumes, for levels 1 to $L$.
6. The natural logarithms of group 5, with each cumulative volume floored at $10^{-8}$ before taking its logarithm.

These first six groups give 78 coordinates for ten levels after historical standardisation. The updated setup adds volume imbalance and imbalance using log cumulative volumes, each in both original and historically standardised form. With B and A denoting cumulative bid and ask volumes through a level, these are `(B-A)/(B+A+1e-6)` and `(log(max(B,1e-8))-log(max(A,1e-8)))/(log(max(B,1e-8))+log(max(A,1e-8))+1e-6)`. The additional 40 coordinates bring the total to 118. These two imbalance measures have distinct column names. Prices are converted back from dollars to LOBSTER integer units before subtraction. Keeping these units matters because historical standardisation adds a fixed constant to the variance.

For each coordinate selected for historical standardisation, let $x_r$ be its value at sampled observation $r$, starting at zero. Historical standardisation uses a mean $\mu_r$ and variance estimate $v_r$ based only on earlier observations. With **Standardisation history** $H$, set $\lambda=0.01^{1/H}$, $\mu_0=x_0$ and $v_0=0$. The transformation and updates are

$$
z_r=\frac{x_r-\mu_r}{\sqrt{v_r+10^{-6}}}.
$$

$$
\mu_{r+1}=\lambda\mu_r+(1-\lambda)x_r.
$$

$$
v_{r+1}=\lambda v_r+(1-\lambda)(x_r-\mu_r)^2.
$$

The first $H$ observations are excluded from PP. Earlier observations continue to contribute with exponentially decreasing weight, so $H$ is not a hard cutoff. Initialisation starts at the selected period and restarts after an invalid sampled snapshot. No earlier data are borrowed from outside the selection. Counts of observations excluded during initialisation are recorded. Clock sampling applies these updates to clock samples, while message sampling applies them to messages. These choices generally give different results.

**Every message** preserves every original row, including separate messages with identical timestamps. **Messages per estimation window** counts the last $M$ original messages before exclusions. **Messages between snapshots** controls spacing when several snapshots form one observation. Invalid messages are not removed before either operation. Every constituent snapshot must remain inside the window, and combined observations cannot cross invalid messages, including those with tied timestamps. Maximum snapshot age does not apply because each message supplies a snapshot at its own time.

PP estimates are still evaluated at regular clock intervals, including all messages at each endpoint. Early estimates without $M$ messages in the selected period are undefined. The same feature formulas also work with clock sampling or other scaling choices, but those combinations do not reproduce the collaborator's historical transformation.

On valid aligned input rows, the feature and standardisation formulas match the repository. The app retains its existing validity checks and correctly reads headerless LOBSTER files. It does not reproduce the repository loader's omission of the first row or the feature functions' omission of the final row. These differences, missing prices and restarts after invalid observations can affect comparisons. The source URL, feature order, units, scaling parameters and exclusion policies are included in exported results.

For a nonconstant cloud with `N` points and `D` coordinates, PP lies between 1 and `min(N - 1, D)`. The chart and inspector show this sample size ceiling. Fewer than two points or an identical point cloud gives an undefined result, displayed as a gap. Normalized dimension is PP divided by the full coordinate count. Variation across levels within a snapshot and variation between snapshots through time describe different quantities. Neither a high PP value nor a low one guarantees a manifold's true geometric dimension.

Each endpoint retains candidate and excluded point counts, status, source metadata, and its singular spectrum. History records additionally report window coverage and invalid or stale interval counts. Summary means use defined endpoints only.

There is no fixed limit on the number of coordinates. Sweeps allow up to 5,000 evaluation times. The calculation budget is 300 billion units of the work estimate `number of estimates × rows × columns × min(rows, columns)`, increased from 200 million. Benchmarks on the local machine with Tom's encoding and five snapshots per row produced 390 columns and took about 8.5 seconds for 1,000 messages per window with estimates every minute, and 12.9 seconds for 5,000 messages per window with estimates every two minutes. A full run through the app with 5,000 messages per window and estimates every minute completed in 30.7 seconds, producing 388 valid estimates from 391 requested. The new allowance targets calculations lasting tens of seconds rather than immediate results. It is an approximate workload allowance, not a guaranteed runtime or a 30 second timeout.

Before allocating the feature matrices, the app estimates additional memory for the sampled features, temporary matrix copies, coordinate names and retained results. Requests estimated above 1 GiB are rejected with guidance. This estimate does not include already loaded data, previous runs, the browser or other applications, so it is not a cap on the app's total memory use. Existing sample limits remain at 100,000 clock observations and 500,000 messages per selected period. Work and memory estimates and the active limits are recorded with each result.

The PP formula, observations and sampling are unchanged. Requests above the remaining limits are rejected rather than silently subsampled. To reduce a large request, calculate dimension less often, shorten the selected period or estimation window, or use fewer levels or snapshots per observation.

### Interpretation and records

Exclusion counts can overlap and should not be added together. Serial dependence and overlapping response horizons mean row counts are not counts of independent observations. No ordinary independent sample confidence intervals, p values, causal claims, or trading performance estimates are produced. Selecting settings after looking at evaluation results makes that evaluation exploratory. Evaluation on the same date does not establish stability across trading days.

Overlapping history windows, delay vectors, and adjacent level patches also contain dependent observations. PP produces no confidence intervals. Changing embeddings, scaling, patch width, history length, or sampling changes the measured object. Compare runs with those settings and point counts visible.

Every downloadable run contains source hashes, source sizes, software versions, source code hashes, method parameters, sample counts, and result arrays. PCA and response records include their split time. Dimension records include all evaluation times and point definitions. All three methods are deterministic and use no random seed. The record identifies the implementation but does not embed its source, so retain these application files alongside important experiment records.

## Local operation

The application binds only to `127.0.0.1`. Debugging, automatic reload, and Dash version checks are disabled. Scripts and styles are served locally. It does not use cloud hosting, remote fonts, analytics, or external data services. Foreign Host and Origin headers and cross site browser requests are rejected. This is a personal local application, not a deployment intended for network sharing. Other software running on the computer may access it.

The application does not write data, cache files, run history, logs, or transformed data to disk. Bytecode writing is disabled. Explicit download controls save files through the browser. Finder launch opens the normal browser, whose own history and storage behavior remain under its settings.

## Recreate the environment

The app uses its own virtual environment. The local environment is excluded from Git. Each recipient must create an environment using the setup instructions below.

For a new installation, create a Python 3.11 virtual environment and install the pinned direct dependencies in `requirements.txt`. These commands create and populate that environment and should only be run when installation is intended.

```sh
python3.11 -m venv lobster_app/.venv
lobster_app/.venv/bin/python -m pip install -r lobster_app/requirements.txt
```

Run the research checks from the `lobster-app` folder.

```sh
lobster_app/.venv/bin/python -B -m unittest lobster_app.test_research -v
```

The tests exercise clock alignment, duplicate timestamps, invalid quotes and halts, stale intervals, training boundary exclusion, training only preprocessing, and synthetic matrix rank. Dimension tests cover the singular value formula, constant clouds, scaling, snapshot patches, delay vectors, and window boundaries. They use in memory fixtures and create no data files.

## Sources

* [LOBSTER output specification](https://php.lobsterdata.com/info/DataStructure.php)
* [LOBSTER reconstruction notes](https://php.lobsterdata.com/info/help_reconstruction.php)
* [LOBSTER FAQ](https://php.lobsterdata.com/info/help_faq_general.php)
* [Patnaik Pearson definition and spectral formula](https://arxiv.org/html/2606.19268v4#S3.SS3)
* [Dash documentation](https://dash.plotly.com/)
