from uldaq import get_daq_device_inventory, DaqDevice, InterfaceType, AiInputMode, Range, AInFlag, ScanStatus, ScanOption, AInScanFlag, DevMemInfo, WaitType, create_float_buffer
import numpy as np


# process of getting signal
def signal_processing(channel_total, srate, stime):
    low_channel = min(channel_total)
    high_channel = max(channel_total)
    num_scan_channels = high_channel - low_channel + 1
    srate = srate
    samples_per_channel = srate * stime
    data_buffer = create_float_buffer(num_scan_channels, samples_per_channel)
    range_index = 0
    descriptor_index = 0
    interface_type = InterfaceType.USB
    flags = AInScanFlag.DEFAULT
    input_mode = AiInputMode.SINGLE_ENDED
    scan_options = ScanOption.DEFAULTIO
    daq_device = None
    ai_device = None
    status = ScanStatus.IDLE
    try:
        # Get descriptors for all of the available DAQ devices.
        devices = get_daq_device_inventory(interface_type)
        number_of_devices = len(devices)
        if number_of_devices == 0:
            raise Exception('Error: No DAQ devices found')
        # Create the DAQ device object associated with the specified descriptor index.
        daq_device = DaqDevice(devices[descriptor_index])
        # Get the AiDevice object and verify that it is valid.
        ai_device = daq_device.get_ai_device()
        if ai_device is None:
            raise Exception(
                'Error: The DAQ device does not support analog input')
        # Verify that the specified device supports hardware pacing for analog input.
        ai_info = ai_device.get_info()
        if not ai_info.has_pacer():
            raise Exception(
                '\nError: The specified DAQ device does not support hardware paced analog input')
        # Establish a connection to the DAQ device.
        descriptor = daq_device.get_descriptor()
        print('\nConnecting to', descriptor.dev_string, '- please wait...')
        daq_device.connect()
        ranges = ai_info.get_ranges(input_mode)
        if range_index >= len(ranges):
            range_index = len(ranges) - 1
        try:
            print('Start sampling...')
            rate = ai_device.a_in_scan(low_channel, high_channel, input_mode, Range.BIP5VOLTS, samples_per_channel,
                                       rate, scan_options, flags, data_buffer)
            print("mem = ", daq_device.get_info().get_mem_info())
            print('data scanning')
            datas = []
            ai_device.scan_wait(WaitType.WAIT_UNTIL_DONE, -1)
            for channel_number in channel_total:
                datas.append(data_buffer[channel_number::num_scan_channels])

        except (ValueError, NameError, SyntaxError):
            pass
    except Exception as e:
        print('\n', e)
    finally:
        # Stop the acquisition if it is still running.
        ai_device.scan_stop()
        daq_device.disconnect()
        daq_device.release()
        return datas
        print("get data finished")


# process of getting feature
def get_feature(data):
    datas = data
    feature_list = ['data_mean', 'data_var', 'data_std', 'data_rms',
                    'data_max', 'data_min', 'data_skew', 'data_kur', 'data_peaktopeak']
    feature = np.zeros(len(feature_list))
    freature_group = []
    for i in range(len(datas)):
        data = np.array(data[i])
        feature[0] = np.mean(data)
        feature[1] = np.var(data)
        feature[2] = np.std(data)
        feature[3] = np.sqrt(np.mean(data**2))
        feature[4] = np.max(data)
        feature[5] = np.min(data)
        feature[6] = np.mean((data - feature[0]) ** 3)
        feature[7] = np.mean((data - feature[0]) ** 4) / pow(feature[1], 2)
        feature[8] = feature[4] - feature[5]
        group = {i: dict(zip(feature_list, feature))}
        freature_group.append(group)
    return freature_group


# main process
def main():
    # set all the channel you needed
    channel_total = [0, 1]
    # set sample rate
    srate = 12800
    # set sample time
    stime = 1
    datas = signal_processing(channel_total, srate, stime)
    # get all the feature of each channel, type is a dict list
    feature = get_feature(datas)
    print(feature)


if __name__ == '__main__':
    try:
        while True:
            try:
                main()
            except KeyboardInterrupt:
                break
    finally:
        pass
