package common

type InferType int

const (
	LocalStorage InferType = 1
	ServerInfer  InferType = 2
)

type URI struct {
	Scheme InferType
	Path   string
}

func ParseURI(uri string) (URI, error) {
	if uri[0:7] == "locstr:" {
		return URI{
			Scheme: LocalStorage,
			Path:   uri[7:],
		}, nil
	}

	return URI{
		Scheme: ServerInfer,
		Path:   uri,
	}, nil
}
