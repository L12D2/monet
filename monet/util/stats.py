import numpy as np


def STDO(obs, mod, axis=None):
    """Standard deviation of Observations.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array (not used in this function).
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the standard deviation.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Standard deviation of observations.
    """
    return np.ma.std(obs, axis=axis)


def STDP(obs, mod, axis=None):
    """Standard deviation of Predictions (model values).

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array (not used in this function).
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the standard deviation.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Standard deviation of model predictions.
    """
    return np.ma.std(mod, axis=axis)


def MNB(obs, mod, axis=None):
    """Mean Normalized Bias (%).

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the mean.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Mean normalized bias in percent.
    """
    return np.ma.masked_invalid((mod - obs) / obs).mean(axis=axis) * 100.0


def MNE(obs, mod, axis=None):
    """Mean Normalized Error (%).

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the mean.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Mean normalized error in percent.
    """
    return np.ma.masked_invalid(np.abs(mod - obs) / obs).mean(axis=axis) * 100.0


def MdnNB(obs, mod, axis=None):
    """Median Normalized Bias (%).

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the median.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Median normalized bias in percent.
    """
    return np.ma.median(np.ma.masked_invalid((mod - obs) / obs), axis=axis) * 100.0


def MdnNE(obs, mod, axis=None):
    """Median Normalized Error (%).

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the median.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Median normalized error in percent.
    """
    return np.ma.median(np.ma.masked_invalid(np.abs(mod - obs) / obs), axis=axis) * 100.0


def NMdnGE(obs, mod, axis=None):
    """Normalized Median Gross Error.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the median.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Normalized median gross error.
    """
    return np.ma.median(np.ma.masked_invalid(np.abs(mod - obs) / obs), axis=axis)


def NO(obs, mod, axis=None):
    """Number of Observations.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array (not used in this function).
    axis : int or tuple of ints, optional
        Axis or axes along which to count.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or int
        Number of valid observation points.
    """
    return np.ma.count(obs, axis=axis)


def NOP(obs, mod, axis=None):
    """Number of Observation-Prediction Pairs.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to count.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or int
        Number of valid observation-prediction pairs.
    """
    return np.ma.count(np.ma.masked_invalid((obs - mod) / obs), axis=axis)


def NP(obs, mod, axis=None):
    """Number of Predictions.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array (not used in this function).
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to count.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or int
        Number of valid model prediction points.
    """
    return np.ma.count(mod, axis=axis)


def MO(obs, mod, axis=None):
    """Mean of Observations.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array (not used in this function).
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the mean.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Mean of observations.
    """
    return np.ma.mean(obs, axis=axis)


def MP(obs, mod, axis=None):
    """Mean of Predictions.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array (not used in this function).
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the mean.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Mean of model predictions.
    """
    return np.ma.mean(mod, axis=axis)


def MdnO(obs, mod, axis=None):
    """Median of Observations.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array (not used in this function).
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the median.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Median of observations.
    """
    return np.ma.median(obs, axis=axis)


def MdnP(obs, mod, axis=None):
    """Median of Predictions.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array (not used in this function).
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the median.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Median of model predictions.
    """
    return np.ma.median(mod, axis=axis)


def RM(obs, mod, axis=None):
    """Ratio of Means (mod/obs).

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the means.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Ratio of means (mod/obs).
    """
    return np.ma.mean(mod, axis=axis) / np.ma.mean(obs, axis=axis)


def RMdn(obs, mod, axis=None):
    """Ratio of Medians (mod/obs).

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the medians.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Ratio of medians (mod/obs).
    """
    return np.ma.median(mod, axis=axis) / np.ma.median(obs, axis=axis)


def MB(obs, mod, axis=None):
    """Mean Bias (mod - obs).

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the mean.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Mean bias.
    """
    return np.ma.mean(mod - obs, axis=axis)


def MdnB(obs, mod, axis=None):
    """Median Bias (mod - obs).

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the median.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Median bias.
    """
    return np.ma.median(mod - obs, axis=axis)


def WDMB_m(obs, mod, axis=None):
    """Wind Direction Mean Bias - Modified.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation wind direction data array in degrees.
    mod : numpy.ndarray
        Model wind direction data array in degrees.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the mean.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Wind direction mean bias (modified calculation).
    """
    d = mod - obs
    d = np.where(d > 180, d - 360, d)
    d = np.where(d < -180, d + 360, d)
    return np.ma.mean(d, axis=axis)


def WDMB(obs, mod, axis=None):
    """Wind Direction Mean Bias.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation wind direction data array in degrees.
    mod : numpy.ndarray
        Model wind direction data array in degrees.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the mean.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Wind direction mean bias.
    """
    d = (mod - obs) % 360
    d = np.where(d > 180, d - 360, d)
    return np.ma.mean(d, axis=axis)


def WDMdnB(obs, mod, axis=None):
    """Wind Direction Median Bias.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation wind direction data array in degrees.
    mod : numpy.ndarray
        Model wind direction data array in degrees.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the median.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Wind direction median bias.
    """
    d = (mod - obs) % 360
    d = np.where(d > 180, d - 360, d)
    return np.ma.median(d, axis=axis)


def NMB(obs, mod, axis=None):
    """Normalized Mean Bias.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the means.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Normalized mean bias.
    """
    return np.ma.sum(mod - obs, axis=axis) / np.ma.sum(obs, axis=axis)


def WDNMB_m(obs, mod, axis=None):
    """Wind Direction Normalized Mean Bias - Modified.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation wind direction data array in degrees.
    mod : numpy.ndarray
        Model wind direction data array in degrees.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the mean.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Wind direction normalized mean bias (modified calculation).
    """
    d = mod - obs
    d = np.where(d > 180, d - 360, d)
    d = np.where(d < -180, d + 360, d)
    return np.ma.sum(d, axis=axis) / np.ma.sum(obs, axis=axis)


def NMB_ABS(obs, mod, axis=None):
    """Normalized Mean Bias using absolute values.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the means.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Normalized mean bias with absolute values.
    """
    return np.ma.sum(mod - obs, axis=axis) / np.ma.sum(np.abs(obs), axis=axis)


def NMdnB(obs, mod, axis=None):
    """Normalized Median Bias.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the median.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Normalized median bias.
    """
    return np.ma.median(mod - obs, axis=axis) / np.ma.median(obs, axis=axis)


def FB(obs, mod, axis=None):
    """Fractional Bias.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the mean.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Fractional bias.
    """
    return np.ma.mean(2.0 * (mod - obs) / (mod + obs), axis=axis)


def ME(obs, mod, axis=None):
    """Mean Error (absolute difference).

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the mean.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Mean error.
    """
    return np.ma.mean(np.abs(mod - obs), axis=axis)


def MdnE(obs, mod, axis=None):
    """Median Error (absolute difference).

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the median.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Median error.
    """
    return np.ma.median(np.abs(mod - obs), axis=axis)


def WDME_m(obs, mod, axis=None):
    """Wind Direction Mean Error - Modified.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation wind direction data array in degrees.
    mod : numpy.ndarray
        Model wind direction data array in degrees.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the mean.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Wind direction mean error (modified calculation).
    """
    d = np.abs(mod - obs)
    d = np.where(d > 180, 360 - d, d)
    return np.ma.mean(d, axis=axis)


def WDME(obs, mod, axis=None):
    """Wind Direction Mean Error.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation wind direction data array in degrees.
    mod : numpy.ndarray
        Model wind direction data array in degrees.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the mean.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Wind direction mean error.
    """
    d = np.abs(mod - obs)
    d = np.minimum(d, 360 - d)
    return np.ma.mean(d, axis=axis)


def WDMdnE(obs, mod, axis=None):
    """Wind Direction Median Error.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation wind direction data array in degrees.
    mod : numpy.ndarray
        Model wind direction data array in degrees.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the median.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Wind direction median error.
    """
    d = np.abs(mod - obs)
    d = np.minimum(d, 360 - d)
    return np.ma.median(d, axis=axis)


def NME_m(obs, mod, axis=None):
    """Normalized Mean Error - Modified.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the sum.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Normalized mean error (modified calculation).
    """
    return np.ma.sum(np.abs(mod - obs), axis=axis) / np.ma.sum(obs, axis=axis)


def NME_m_ABS(obs, mod, axis=None):
    """Normalized Mean Error - Modified, using absolute values.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the sum.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Normalized mean error using absolute values (modified calculation).
    """
    return np.ma.sum(np.abs(mod - obs), axis=axis) / np.ma.sum(np.abs(obs), axis=axis)


def NME(obs, mod, axis=None):
    """Normalized Mean Error.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the mean.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Normalized mean error.
    """
    return np.ma.mean(np.abs(mod - obs) / obs, axis=axis)


def NMdnE(obs, mod, axis=None):
    """Normalized Median Error.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the median.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Normalized median error.
    """
    return np.ma.median(np.abs(mod - obs), axis=axis) / np.ma.median(obs, axis=axis)


def FE(obs, mod, axis=None):
    """Fractional Error.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the mean.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Fractional error.
    """
    return np.ma.mean(2.0 * np.abs(mod - obs) / (np.abs(mod) + np.abs(obs)), axis=axis)


def USUTPB(obs, mod, axis=None):
    """Unpaired Space/Unpaired Time Peak Bias.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the maximum.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Unpaired space/unpaired time peak bias.
    """
    return np.ma.max(mod, axis=axis) / np.ma.max(obs, axis=axis) - 1.0


def USUTPE(obs, mod, axis=None):
    """Unpaired Space/Unpaired Time Peak Error.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the maximum.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Unpaired space/unpaired time peak error.
    """
    return np.abs(np.ma.max(mod, axis=axis) / np.ma.max(obs, axis=axis) - 1.0)


def MNPB(obs, mod, paxis, axis=None):
    """Mean Normalized Peak Bias.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    paxis : int or tuple of ints
        Axis or axes along which to calculate the peak (maximum).
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the mean.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Mean normalized peak bias.
    """
    return np.ma.mean(
        np.ma.max(mod, axis=paxis) / np.ma.max(obs, axis=paxis) - 1.0, axis=axis
    )


def MdnNPB(obs, mod, paxis, axis=None):
    """Median Normalized Peak Bias.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    paxis : int or tuple of ints
        Axis or axes along which to calculate the peak (maximum).
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the median.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Median normalized peak bias.
    """
    return np.ma.median(
        np.ma.max(mod, axis=paxis) / np.ma.max(obs, axis=paxis) - 1.0, axis=axis
    )


def MNPE(obs, mod, paxis, axis=None):
    """Mean Normalized Peak Error.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    paxis : int or tuple of ints
        Axis or axes along which to calculate the peak (maximum).
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the mean.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Mean normalized peak error.
    """
    return np.ma.mean(
        np.abs(np.ma.max(mod, axis=paxis) / np.ma.max(obs, axis=paxis) - 1.0), axis=axis
    )


def MdnNPE(obs, mod, paxis, axis=None):
    """Median Normalized Peak Error.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    paxis : int or tuple of ints
        Axis or axes along which to calculate the peak (maximum).
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the median.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Median normalized peak error.
    """
    return np.ma.median(
        np.abs(np.ma.max(mod, axis=paxis) / np.ma.max(obs, axis=paxis) - 1.0), axis=axis
    )


def NMPB(obs, mod, paxis, axis=None):
    """Normalized Mean Peak Bias.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    paxis : int or tuple of ints
        Axis or axes along which to calculate the peak (maximum).
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the sum.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Normalized mean peak bias.
    """
    return (
        np.ma.sum(np.ma.max(mod, axis=paxis) - np.ma.max(obs, axis=paxis), axis=axis)
        / np.ma.sum(np.ma.max(obs, axis=paxis), axis=axis)
    )


def NMdnPB(obs, mod, paxis, axis=None):
    """Normalized Median Peak Bias.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    paxis : int or tuple of ints
        Axis or axes along which to calculate the peak (maximum).
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the median.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Normalized median peak bias.
    """
    return (
        np.ma.median(np.ma.max(mod, axis=paxis) - np.ma.max(obs, axis=paxis), axis=axis)
        / np.ma.median(np.ma.max(obs, axis=paxis), axis=axis)
    )


def NMPE(obs, mod, paxis, axis=None):
    """Normalized Mean Peak Error.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    paxis : int or tuple of ints
        Axis or axes along which to calculate the peak (maximum).
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the sum.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Normalized mean peak error.
    """
    return (
        np.ma.sum(np.abs(np.ma.max(mod, axis=paxis) - np.ma.max(obs, axis=paxis)), axis=axis)
        / np.ma.sum(np.ma.max(obs, axis=paxis), axis=axis)
    )


def NMdnPE(obs, mod, paxis, axis=None):
    """Normalized Median Peak Error.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    paxis : int or tuple of ints
        Axis or axes along which to calculate the peak (maximum).
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the median.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Normalized median peak error.
    """
    return (
        np.ma.median(np.abs(np.ma.max(mod, axis=paxis) - np.ma.max(obs, axis=paxis)), axis=axis)
        / np.ma.median(np.ma.max(obs, axis=paxis), axis=axis)
    )


def PSUTMNPB(obs, mod, axis=None):
    """Paired Space/Unpaired Time Mean Normalized Peak Bias.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the mean.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Paired space/unpaired time mean normalized peak bias.
    """
    return np.ma.mean(np.ma.max(mod, axis=axis) / np.ma.max(obs, axis=axis) - 1.0)


def PSUTMdnNPB(obs, mod, axis=None):
    """Paired Space/Unpaired Time Median Normalized Peak Bias.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the median.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Paired space/unpaired time median normalized peak bias.
    """
    return np.ma.median(np.ma.max(mod, axis=axis) / np.ma.max(obs, axis=axis) - 1.0)


def PSUTMNPE(obs, mod, axis=None):
    """Paired Space/Unpaired Time Mean Normalized Peak Error.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the mean.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Paired space/unpaired time mean normalized peak error.
    """
    return np.ma.mean(np.abs(np.ma.max(mod, axis=axis) / np.ma.max(obs, axis=axis) - 1.0))


def PSUTMdnNPE(obs, mod, axis=None):
    """Paired Space/Unpaired Time Median Normalized Peak Error.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the median.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Paired space/unpaired time median normalized peak error.
    """
    return np.ma.median(np.abs(np.ma.max(mod, axis=axis) / np.ma.max(obs, axis=axis) - 1.0))


def PSUTNMPB(obs, mod, axis=None):
    """Paired Space/Unpaired Time Normalized Mean Peak Bias.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the sum.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Paired space/unpaired time normalized mean peak bias.
    """
    return (
        np.ma.sum(np.ma.max(mod, axis=axis) - np.ma.max(obs, axis=axis))
        / np.ma.sum(np.ma.max(obs, axis=axis))
    )


def PSUTNMPE(obs, mod, axis=None):
    """Paired Space/Unpaired Time Normalized Mean Peak Error.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the sum.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Paired space/unpaired time normalized mean peak error.
    """
    return (
        np.ma.sum(np.abs(np.ma.max(mod, axis=axis) - np.ma.max(obs, axis=axis)))
        / np.ma.sum(np.ma.max(obs, axis=axis))
    )


def PSUTNMdnPB(obs, mod, axis=None):
    """Paired Space/Unpaired Time Normalized Median Peak Bias.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the median.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Paired space/unpaired time normalized median peak bias.
    """
    return (
        np.ma.median(np.ma.max(mod, axis=axis) - np.ma.max(obs, axis=axis))
        / np.ma.median(np.ma.max(obs, axis=axis))
    )


def PSUTNMdnPE(obs, mod, axis=None):
    """Paired Space/Unpaired Time Normalized Median Peak Error.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the median.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Paired space/unpaired time normalized median peak error.
    """
    return (
        np.ma.median(np.abs(np.ma.max(mod, axis=axis) - np.ma.max(obs, axis=axis)))
        / np.ma.median(np.ma.max(obs, axis=axis))
    )


def R2(obs, mod, axis=None):
    """Coefficient of determination (r-squared).

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the correlation.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Coefficient of determination (r-squared).
    """
    return np.ma.corrcoef(obs, mod, rowvar=False)[0, 1] ** 2


def RMSE(obs, mod, axis=None):
    """Root Mean Square Error.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the mean.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Root mean square error.
    """
    return np.ma.sqrt(np.ma.mean((mod - obs) ** 2, axis=axis))


def WDRMSE_m(obs, mod, axis=None):
    """Wind Direction Root Mean Square Error - Modified.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation wind direction data array in degrees.
    mod : numpy.ndarray
        Model wind direction data array in degrees.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the mean.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Wind direction root mean square error (modified calculation).
    """
    d = mod - obs
    d = np.where(d > 180, d - 360, d)
    d = np.where(d < -180, d + 360, d)
    return np.ma.sqrt(np.ma.mean(d ** 2, axis=axis))


def WDRMSE(obs, mod, axis=None):
    """Wind Direction Root Mean Square Error.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation wind direction data array in degrees.
    mod : numpy.ndarray
        Model wind direction data array in degrees.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the mean.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Wind direction root mean square error.
    """
    d = (mod - obs) % 360
    d = np.where(d > 180, d - 360, d)
    return np.ma.sqrt(np.ma.mean(d ** 2, axis=axis))


def RMSEs(obs, mod, axis=None):
    """Systematic Root Mean Square Error.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the mean.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Systematic root mean square error.
    """
    import statsmodels.api as sm

    a1, a0, r, p, std_err = sm.OLS(mod.ravel(), sm.add_constant(obs.ravel())).fit().params
    p = a0 + a1 * obs
    return np.ma.sqrt(np.ma.mean((p - obs) ** 2, axis=axis))


def matchmasks(a1, a2):
    """Match masks between two masked arrays.

    Parameters
    ----------
    a1 : numpy.ndarray or numpy.ma.MaskedArray
        First array.
    a2 : numpy.ndarray or numpy.ma.MaskedArray
        Second array.

    Returns
    -------
    tuple of numpy.ma.MaskedArray
        Tuple of (a1, a2) with matched masks.
    """
    # Convert regular arrays to masked arrays if needed
    if not isinstance(a1, np.ma.MaskedArray):
        a1 = np.ma.array(a1)
    if not isinstance(a2, np.ma.MaskedArray):
        a2 = np.ma.array(a2)

    if np.any(a1.mask) or np.any(a2.mask):
        mask = np.logical_or(a1.mask, a2.mask)
        a1 = np.ma.masked_array(a1.data, mask=mask)
        a2 = np.ma.masked_array(a2.data, mask=mask)
    return a1, a2


def matchedcompressed(a1, a2):
    """Return compressed arrays (no masked values) with matched masks.

    Parameters
    ----------
    a1 : numpy.ma.MaskedArray
        First masked array.
    a2 : numpy.ma.MaskedArray
        Second masked array.

    Returns
    -------
    tuple of numpy.ndarray
        Tuple of (a1_compressed, a2_compressed) with no masked values.
    """
    a1, a2 = matchmasks(a1, a2)
    return a1.compressed(), a2.compressed()


def RMSEu(obs, mod, axis=None):
    """Unsystematic Root Mean Square Error.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the mean.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Unsystematic root mean square error.
    """
    import statsmodels.api as sm

    a1, a0, r, p, std_err = sm.OLS(mod.ravel(), sm.add_constant(obs.ravel())).fit().params
    p = a0 + a1 * obs
    return np.ma.sqrt(np.ma.mean((mod - p) ** 2, axis=axis))


def d1(obs, mod, axis=None):
    """Index of Agreement, d1, modified version of Willmott (1982).

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the sums.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Index of agreement, d1.
    """
    return 1.0 - (
        np.ma.sum(np.abs(mod - obs), axis=axis)
        / np.ma.sum(np.abs(mod - np.ma.mean(obs, axis=axis)) + np.abs(obs - np.ma.mean(obs, axis=axis)), axis=axis)
    )


def E1(obs, mod, axis=None):
    """Nash-Sutcliffe coefficient of efficiency.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the sums.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Nash-Sutcliffe coefficient of efficiency.
    """
    return (
        1.0
        - np.ma.sum((mod - obs) ** 2, axis=axis)
        / np.ma.sum((obs - np.ma.mean(obs, axis=axis)) ** 2, axis=axis)
    )


def IOA_m(obs, mod, axis=None):
    """Index of Agreement - Modified.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the sums.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Index of agreement (modified calculation).
    """
    obsmean = obs.mean(axis=axis)
    if axis is not None:
        ax = list(np.arange(obs.ndim))
        ax.remove(axis)
        ax = tuple(ax)
        denom = np.ma.sum(
            (np.abs(mod - obsmean) + np.abs(obs - obsmean)),
            axis=ax,
        )
    else:
        denom = np.ma.sum(np.abs(mod - obsmean) + np.abs(obs - obsmean))
    return 1.0 - (
        np.ma.sum(np.abs(mod - obs), axis=axis) / denom
    )


def IOA(obs, mod, axis=None):
    """Index of Agreement.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the sums.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Index of agreement.
    """
    obsmean = obs.mean(axis=axis)
    if axis is not None:
        ax = list(np.arange(obs.ndim))
        ax.remove(axis)
        ax = tuple(ax)
        denom = np.ma.sum(
            (np.abs(mod - obsmean) + np.abs(obs - obsmean)) ** 2,
            axis=ax,
        )
    else:
        denom = np.ma.sum((np.abs(mod - obsmean) + np.abs(obs - obsmean)) ** 2)
    return 1.0 - (
        np.ma.sum((mod - obs) ** 2, axis=axis) / denom
    )


def circlebias_m(b):
    """Circular Bias - Modified.

    Parameters
    ----------
    b : numpy.ndarray
        Array of bias values in degrees.

    Returns
    -------
    numpy.ndarray
        Modified circular bias in degrees.
    """
    b = np.where(b > 180, b - 360, b)
    b = np.where(b < -180, b + 360, b)
    return b


def circlebias(b):
    """Circular Bias.

    Parameters
    ----------
    b : numpy.ndarray
        Array of bias values in degrees.

    Returns
    -------
    numpy.ndarray
        Circular bias in degrees.
    """
    b = b % 360
    b = np.where(b > 180, b - 360, b)
    return b


def WDIOA_m(obs, mod, axis=None):
    """Wind Direction Index of Agreement - Modified.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation wind direction data array in degrees.
    mod : numpy.ndarray
        Model wind direction data array in degrees.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the sums.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Wind direction index of agreement (modified calculation).
    """
    d = mod - obs
    d = circlebias_m(d)
    obsmean = obs.mean(axis=axis)
    if axis is not None:
        ax = list(np.arange(obs.ndim))
        ax.remove(axis)
        ax = tuple(ax)
        denom = np.ma.sum(
            (np.abs(mod - obsmean) + np.abs(obs - obsmean)),
            axis=ax,
        )
    else:
        denom = np.ma.sum(np.abs(mod - obsmean) + np.abs(obs - obsmean))
    return 1.0 - (
        np.ma.sum(np.abs(d), axis=axis) / denom
    )


def WDIOA(obs, mod, axis=None):
    """Wind Direction Index of Agreement.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation wind direction data array in degrees.
    mod : numpy.ndarray
        Model wind direction data array in degrees.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the sums.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Wind direction index of agreement.
    """
    d = mod - obs
    d = circlebias(d)
    obsmean = obs.mean(axis=axis)
    if axis is not None:
        ax = list(np.arange(obs.ndim))
        ax.remove(axis)
        ax = tuple(ax)
        denom = np.ma.sum(
            (np.abs(mod - obsmean) + np.abs(obs - obsmean)) ** 2,
            axis=ax,
        )
    else:
        denom = np.ma.sum((np.abs(mod - obsmean) + np.abs(obs - obsmean)) ** 2)
    return 1.0 - (
        np.ma.sum(d ** 2, axis=axis) / denom
    )


def AC(obs, mod, axis=None):
    """Anomaly Correlation.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the means and sums.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Anomaly correlation.
    """
    obsbar = np.ma.mean(obs, axis=axis)
    modbar = np.ma.mean(mod, axis=axis)
    top = np.ma.sum((obs - obsbar) * (mod - modbar), axis=axis)
    bot = np.sqrt(
        np.ma.sum((obs - obsbar) ** 2, axis=axis) * np.ma.sum((mod - modbar) ** 2, axis=axis)
    )
    return top / bot


def WDAC(obs, mod, axis=None):
    """Wind Direction Anomaly Correlation.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation wind direction data array in degrees.
    mod : numpy.ndarray
        Model wind direction data array in degrees.
    axis : int or tuple of ints, optional
        Axis or axes along which to calculate the means and sums.
        If None, compute over the whole array.

    Returns
    -------
    numpy.ndarray or float
        Wind direction anomaly correlation.
    """
    obsbar = np.ma.mean(obs, axis=axis)
    modbar = np.ma.mean(mod, axis=axis)
    o = circlebias(obs - obsbar)
    m = circlebias(mod - modbar)
    top = np.ma.sum(o * m, axis=axis)
    bot = np.sqrt(np.ma.sum(o ** 2, axis=axis) * np.ma.sum(m ** 2, axis=axis))
    return top / bot


def HSS(obs, mod, minval, maxval):
    """Heidke Skill Score.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    minval : float
        Minimum value for hit/miss categorization.
    maxval : float
        Maximum value for hit/miss categorization.

    Returns
    -------
    float
        Heidke skill score.
    """
    # need to iterate through al threshold values
    hits = misses = false_alarms = correct_negatives = 0
    obs, mod = matchedcompressed(obs, mod)
    for i in range(len(obs)):
        if (obs[i] >= minval) and (obs[i] <= maxval) and (mod[i] >= minval) and (mod[i] <= maxval):
            hits += 1  # hit
        elif ((obs[i] < minval) or (obs[i] > maxval)) and (
            (mod[i] < minval) or (mod[i] > maxval)
        ):
            correct_negatives += 1  # correct negative
        elif ((obs[i] < minval) or (obs[i] > maxval)) and (
            (mod[i] >= minval) and (mod[i] <= maxval)
        ):
            false_alarms += 1  # false alarm
        else:
            misses += 1  # miss
    try:
        HSS = (
            2.0
            * (hits * correct_negatives - misses * false_alarms)
            / (
                (hits + misses) * (misses + correct_negatives)
                + (hits + false_alarms) * (false_alarms + correct_negatives)
            )
        )
    except ZeroDivisionError:
        HSS = None
    return HSS


def ETS(obs, mod, minval, maxval):
    """Equitable Threat Score.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    minval : float
        Minimum value for hit/miss categorization.
    maxval : float
        Maximum value for hit/miss categorization.

    Returns
    -------
    float
        Equitable threat score.
    """
    hits = misses = false_alarms = correct_negatives = 0
    obs, mod = matchedcompressed(obs, mod)
    for i in range(len(obs)):
        if (obs[i] >= minval) and (obs[i] <= maxval) and (mod[i] >= minval) and (mod[i] <= maxval):
            hits += 1  # hit
        elif ((obs[i] < minval) or (obs[i] > maxval)) and (
            (mod[i] < minval) or (mod[i] > maxval)
        ):
            correct_negatives += 1  # correct negative
        elif ((obs[i] < minval) or (obs[i] > maxval)) and (
            (mod[i] >= minval) and (mod[i] <= maxval)
        ):
            false_alarms += 1  # false alarm
        else:
            misses += 1  # miss
    try:
        ar = float(hits + false_alarms) * float(hits + misses) / float(
            hits + misses + false_alarms + correct_negatives
        )
        ETS = (hits - ar) / (hits + misses + false_alarms - ar)
    except ZeroDivisionError:
        ETS = None
    return ETS


def CSI(obs, mod, minval, maxval):
    """Critical Success Index.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    minval : float
        Minimum value for hit/miss categorization.
    maxval : float
        Maximum value for hit/miss categorization.

    Returns
    -------
    float
        Critical success index.
    """
    hits = misses = false_alarms = correct_negatives = 0
    obs, mod = matchedcompressed(obs, mod)
    for i in range(len(obs)):
        if (obs[i] >= minval) and (obs[i] <= maxval) and (mod[i] >= minval) and (mod[i] <= maxval):
            hits += 1  # hit
        elif ((obs[i] < minval) or (obs[i] > maxval)) and (
            (mod[i] < minval) or (mod[i] > maxval)
        ):
            correct_negatives += 1  # correct negative
        elif ((obs[i] < minval) or (obs[i] > maxval)) and (
            (mod[i] >= minval) and (mod[i] <= maxval)
        ):
            false_alarms += 1  # false alarm
        else:
            misses += 1  # miss
    try:
        CSI = hits / (hits + misses + false_alarms)
    except ZeroDivisionError:
        CSI = None
    return CSI


def scores(obs, mod, minval, maxval=1.0e5):
    """Calculate all categorical scores.

    Parameters
    ----------
    obs : numpy.ndarray
        Observation data array.
    mod : numpy.ndarray
        Model data array.
    minval : float
        Minimum value for hit/miss categorization.
    maxval : float, optional
        Maximum value for hit/miss categorization (default: 1.0e5).

    Returns
    -------
    tuple
        Tuple containing (HSS, ETS, CSI) categorical scores.
    """
    h = HSS(obs, mod, minval, maxval)
    e = ETS(obs, mod, minval, maxval)
    c = CSI(obs, mod, minval, maxval)
    return h, e, c


def stats(df, minval, maxval):
    """Calculate standard statistics for observation-model comparisons.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'Obs' and 'CMAQ' columns for comparison.
    minval : float
        Minimum value for categorical scores.
    maxval : float
        Maximum value for categorical scores.

    Returns
    -------
    dict
        Dictionary containing various statistical measures.
    """
    obs = df.Obs.values
    mod = df.CMAQ.values
    svr = {}
    svr["hss"] = HSS(obs, mod, minval, maxval)
    svr["ets"] = ETS(obs, mod, minval, maxval)
    svr["csi"] = CSI(obs, mod, minval, maxval)
    svr["nobs"] = df.Obs.count()
    svr["rmse"] = np.sqrt(((df.CMAQ - df.Obs) ** 2).mean())
    svr["me"] = (df.CMAQ - df.Obs).mean()
    svr["mb"] = (df.CMAQ - df.Obs).mean() / df.Obs.mean()
    svr["ioa"] = 1.0 - (
        (((df.CMAQ - df.Obs) ** 2).sum())
        / ((np.abs(df.CMAQ - df.Obs.mean()) + np.abs(df.Obs - df.Obs.mean())) ** 2).sum()
    )
    svr["corr"] = df.CMAQ.corr(df.Obs)
    return svr
