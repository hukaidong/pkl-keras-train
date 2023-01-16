require "matrix"
require "descriptive_statistics"

class Numeric
  def squared
    self ** 2
  end
end

def parse_matrix(filename)
  data = File.readlines(filename).map do |line|
    line.split.map &:to_f
  end
  Matrix[*data]
end

gt = parse_matrix("gt.npy")
pd = parse_matrix("pd.npy")

pp (gt-pd).collect(&:abs).mean
(gt-pd).column_vectors.each do |col|
  printf "%.2f%%\n", col.map(&:abs).mean/0.02
end
