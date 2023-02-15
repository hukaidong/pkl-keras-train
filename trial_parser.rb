require 'csv'
require "json"
require 'tqdm'

all_trails = Dir['untitled_project/**/trial.json']

datas = all_trails.sort.each.tqdm.map do |file|
  trail = JSON.load_file(file, allow_nan: true)
  data = {}
  data["trial_id"] = trail['trial_id']
  data.update(trail["hyperparameters"]["values"])
  data["score"] = trail["score"]
  data.transform_keys! do |key|
    key.gsub("tuner/", "")
       .gsub("-", "_")
  end
  data
end

CSV.open("trials.csv", "w") do |csv|
  csv << datas.first.keys
  datas.sort_by{|x|x['trial_id']}.each {|d| csv << d.values}
end
