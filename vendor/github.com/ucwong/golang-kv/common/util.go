package common

func SafeCopy(des, src []byte) []byte {
	return append(des[:0], src...)
}
