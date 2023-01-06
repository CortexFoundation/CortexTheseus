package filecache

import (
	"os"
	"time"

	"github.com/anacrolix/missinggo/v2"
)

type itemState struct {
	Accessed time.Time
	Size     int64
}

func (i *itemState) FromOSFileInfo(fi os.FileInfo) {
	i.Size = fi.Size()
	i.Accessed = missinggo.FileInfoAccessTime(fi)
	if fi.ModTime().After(i.Accessed) {
		i.Accessed = fi.ModTime()
	}
}
