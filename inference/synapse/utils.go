package synapse

func ReadImage(inputFilePath string) ([]byte, error) {
	r, rerr := NewFileReader(inputFilePath)
	if rerr != nil {
		return nil, rerr
	}

	// Infer data must between [0, 127)
	data, derr := r.GetBytes()
	if derr != nil {
		return nil, derr
	}

	for i, v := range data {
		data[i] = uint8(v) / 2
	}

	// Tmp Code
	// DumpToFile("tmp.dump", data)

	return data, nil
}
