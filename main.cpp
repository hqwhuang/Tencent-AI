#include <bits/stdc++.h>

using namespace std;

#define INF 2139062143
#define LINF 9187201950435737471
#define mem(a,v) memset(a,v,sizeof(a))
#define maxn 100050
#define rep(i, a, b) for (int i = (a); i <= (b); ++i)
#define red(i, a, b) for (int i = (a); i >= (b); --i)
#define pb push_back
#define eps 1e-6
#define ll long long
#define max(a,b) (a>b?a:b)
#define min(a,b) (a<b?a:b)
typedef pair<int, int> pp;

int batch_size = 250000;
//int batch_size = 25;
const int AGE_NUM = 10;
const int GENDER_NUM = 2;
const int SEQUENCE_LEN = 20;

struct user_profile {
    int age, gender;
    vector<int> creative_ids;
    vector<int> ad_ids;
    vector<int> product_ids;
    vector<int> product_categories;
    vector<int> advertiser_ids;
    vector<int> industries;

    set<int> creative_id_set;
    set<int> ad_id_set;
    set<int> product_id_set;
    set<int> product_category_set;
    set<int> advertiser_id_set;
    set<int> industry_set;

    void add_creative_id(int creative_id) {
        if (creative_id_set.find(creative_id) == creative_id_set.end()) {
            creative_ids.pb(creative_id);
            creative_id_set.insert(creative_id);
        }
    }
    void add_ad_id(int ad_id) {
        if (ad_id_set.find(ad_id) == ad_id_set.end()) {
            ad_ids.pb(ad_id);
            ad_id_set.insert(ad_id);
        }
    }
    void add_product_id(int product_id) {
        if (product_id_set.find(product_id) == product_id_set.end()) {
            product_ids.pb(product_id);
            product_id_set.insert(product_id);
        }
    }
    void add_product_category(int product_category) {
        if (product_category_set.find(product_category) == product_category_set.end()) {
            product_categories.pb(product_category);
            product_category_set.insert(product_category);
        }
    }
    void add_advertiser_id(int advertiser_id) {
        if (advertiser_id_set.find(advertiser_id) == advertiser_id_set.end()) {
            advertiser_ids.pb(advertiser_id);
            advertiser_id_set.insert(advertiser_id);
        }
    }
    void add_industry(int industry) {
        if (industry_set.find(industry) == industry_set.end()) {
            industries.pb(industry);
            industry_set.insert(industry);
        }
    }
    vector<int> get_creative_ids() {
        if (creative_ids.size() <= SEQUENCE_LEN) {
            return creative_ids;
        }
        return vector<int>(creative_ids.rbegin(), creative_ids.rbegin()+SEQUENCE_LEN);
    }
    vector<int> get_ad_ids() {
        if (ad_ids.size() <= SEQUENCE_LEN) {
            return ad_ids;
        }
        return vector<int>(ad_ids.rbegin(), ad_ids.rbegin()+SEQUENCE_LEN);
    }
    vector<int> get_product_ids() {
        if (product_ids.size() <= SEQUENCE_LEN) {
            return product_ids;
        }
        return vector<int>(product_ids.rbegin(), product_ids.rbegin()+SEQUENCE_LEN);
    }
    vector<int> get_product_categories() {
        if (product_categories.size() <= SEQUENCE_LEN) {
            return product_categories;
        }
        return vector<int>(product_categories.rbegin(), product_categories.rbegin()+SEQUENCE_LEN);
    }
    vector<int> get_advertiser_ids() {
        if (advertiser_ids.size() <= SEQUENCE_LEN) {
            return advertiser_ids;
        }
        return vector<int>(advertiser_ids.rbegin(), advertiser_ids.rbegin()+SEQUENCE_LEN);
    }
    vector<int> get_industries() {
        if (industries.size() <= SEQUENCE_LEN) {
            return industries;
        }
        return vector<int>(industries.rbegin(), industries.rbegin()+SEQUENCE_LEN);
    }

    void absorb(const user_profile& a) {
        for(auto v : a.creative_ids) {
            add_creative_id(v);
        }
        for(auto v : a.ad_ids) {
            add_ad_id(v);
        }
        for(auto v : a.product_ids) {
            add_product_id(v);
        }
        for(auto v : a.product_categories) {
            add_product_category(v);
        }
        for(auto v : a.advertiser_ids) {
            add_advertiser_id(v);
        }
        for(auto v : a.industries) {
            add_industry(v);
        }
    }
};

struct creative_profile {
    vector<int> ad_ids;
    vector<int> product_ids;
    vector<int> product_categories;
    vector<int> advertiser_ids;
    vector<int> industries;
    int age_stat[11];
    int gender_stat[3];
    int pv;
    int time_age_stat[100][11];
    int time_gender_stat[100][3];
    int time_pv[100];

    creative_profile(): pv(0) {
        mem(age_stat, 0);
        mem(gender_stat, 0);
        mem(time_age_stat, 0);
        mem(time_gender_stat, 0);
    };

    void add_ad_ids(int aid) {
        ad_ids.pb(aid);
    }

    void add_product_id(int product_id) {
        product_ids.pb(product_id);
    }

    void add_product_category(int pc) {
        product_categories.pb(pc);
    }

    void add_advertiser_id(int aid) {
        advertiser_ids.pb(aid);
    }

    void add_industry(int iid) {
        industries.pb(iid);
    }

    int get_age_ratio(int value, int time) {
        if (time_pv[time] == 0) {
            return 0;
        }
        return (int)(100.0*time_age_stat[time][value]/time_pv[time]);
    }

    int get_gender_ratio(int value, int time) {
        if (time_pv[time] == 0) {
            return 0;
        }
        return (int)(100.0*time_gender_stat[time][value]/time_pv[time]);
    }

    void absorb(const creative_profile& a) {
        for (int i = 1; i < AGE_NUM; i++) {
            age_stat[i] += a.age_stat[i];
        }
        for (int i = 1; i < GENDER_NUM; i++) {
            gender_stat[i] += a.gender_stat[i];
        }
        pv += a.pv;
    }

    void snapshot(int time) {
        for (int i = 1; i < AGE_NUM; i++) {
            time_age_stat[time][i] = age_stat[i];
        }
        for (int i = 1; i < GENDER_NUM; i++) {
            time_gender_stat[time][i] = gender_stat[i];
        }
        time_pv[time] = pv;
    }
};

struct act_log {
    int time, user_id, creative_id, click_times;
};

map<int, user_profile> user_map;
map<int, creative_profile> creative_map;
map<int, user_profile> user_map_cache;
map<int, creative_profile> creative_map_cache;
vector<map<string, vector<int>>> record;
map<string, vector<int>> tmp;
vector<int> tmpv;
set<int> tmps;
act_log log_list[41000000];
set<int> train_creative_id;
set<int> train_ad_id;
set<int> train_product_id;
set<int> train_product_category;
set<int> train_advertiser_id;
set<int> train_industry;
int indexx = 1;

void filetotfrecord(int index, bool testing);
void gen_second_phase(bool testing);

void gen_training_data(bool testing) {
    cout<<"begin to fullfill record"<<endl;
    user_map.clear();
    creative_map.clear();
    user_map_cache.clear();
    creative_map_cache.clear();
    record.clear();
    ifstream user("/Users/huangqingwei/Documents/C++ workspace/codeforces/train_preliminary/user.csv");
    ifstream ad("/Users/huangqingwei/Documents/C++ workspace/codeforces/train_preliminary/adz.csv");
    ifstream click_log("/Users/huangqingwei/Documents/C++ workspace/codeforces/train_preliminary/click_log.csv");

    //read_user_map
    cout<<"begin to read user_map"<<endl;
    string line;
    getline(user, line);
    int uid,age,gender;
    while(getline(user, line)) {
        sscanf(line.c_str(), "%d,%d,%d", &uid, &age, &gender);
        user_map.insert({uid, {age, gender, tmpv, tmpv, tmpv, tmpv, tmpv, tmpv, tmps, tmps, tmps, tmps, tmps, tmps}});
    }
    //read_ad_map
    cout<<"begin to read ad_map"<<endl;
    getline(ad, line);
    int creative_id, ad_id, product_id, product_category, advertiser_id, industry;
    while(getline(ad, line)) {
        sscanf(line.c_str(), "%d,%d,%d,%d,%d,%d", &creative_id, &ad_id, &product_id, &product_category, &advertiser_id, &industry);
        if (creative_map.find(creative_id) == creative_map.end()) {
            creative_map[creative_id] = creative_profile();
        }
        train_creative_id.insert(creative_id);
        train_ad_id.insert(ad_id);
        train_product_id.insert(product_id);
        train_product_category.insert(product_category);
        train_advertiser_id.insert(advertiser_id);
        train_industry.insert(industry);
        creative_map[creative_id].add_ad_ids(ad_id);
        creative_map[creative_id].add_product_id(product_id);
        creative_map[creative_id].add_product_category(product_category);
        creative_map[creative_id].add_advertiser_id(advertiser_id);
        creative_map[creative_id].add_industry(industry);
    }
    //calculate for stat feature
    cout<<"begin to calculate for stat feature"<<endl;
    getline(click_log, line);
    int time, user_id, click_times;
    int count = 0;
    while(getline(click_log, line)) {
        sscanf(line.c_str(), "%d,%d,%d,%d", &time, &user_id, &creative_id, &click_times);
        if (creative_map.find(creative_id) == creative_map.end()) {
            creative_map[creative_id] = creative_profile();
        }
        log_list[count] = {time, user_id, creative_id, click_times};

        count++;
    }
    cout<<"number of log: "<<count<<endl;
//    if (testing)
//        return;
    cout<<"begin to generate record vector"<<endl;
    int tot = count;
    sort(log_list, log_list+tot, [](act_log &a, act_log &b){
        if(a.time == b.time)
            return a.creative_id < b.creative_id;
        return a.time < b.time;
    });
    count = 0;
    int current_time = 1;
    for(int h = 0; h < tot; h++) {
        time = log_list[h].time;
        user_id = log_list[h].user_id;
        creative_id = log_list[h].creative_id;
        click_times = log_list[h].click_times;
        if (current_time != time) {
            for(auto v : user_map_cache) {
                user_map[v.first].absorb(v.second);
            }
            user_map_cache.clear();
            for(auto v : creative_map_cache) {
                creative_map[v.first].absorb(v.second);
                creative_map[v.first].snapshot(current_time);
            }
            creative_map_cache.clear();
            current_time = time;
        }


        creative_map_cache[creative_id].age_stat[user_map[user_id].age] += 1;
        creative_map_cache[creative_id].gender_stat[user_map[user_id].gender] += 1;
        creative_map_cache[creative_id].pv += 1;

        if (testing) {
            continue;
        }

        user_map_cache[user_id].add_creative_id(creative_id);
        user_map_cache[user_id].add_ad_id(creative_map[creative_id].ad_ids[0]);
        user_map_cache[user_id].add_product_id(creative_map[creative_id].product_ids[0]);
        user_map_cache[user_id].add_product_category(creative_map[creative_id].product_categories[0]);
        user_map_cache[user_id].add_advertiser_id(creative_map[creative_id].advertiser_ids[0]);
        user_map_cache[user_id].add_industry(creative_map[creative_id].industries[0]);

        tmp.clear();
        tmp["time"] = vector<int>(1, time);
        tmp["user_id"] = vector<int>(1, user_id);
        tmp["creative_id"] = vector<int>(1, creative_id);
        tmp["age"] = vector<int>(1, user_map[user_id].age);
        tmp["gender"] = vector<int>(1, user_map[user_id].gender);
        tmp["ad_id"] = creative_map[creative_id].ad_ids;
        tmp["product_id"] = creative_map[creative_id].product_ids;
        tmp["product_category"] = creative_map[creative_id].product_categories;
        tmp["advertiser_id"] = creative_map[creative_id].advertiser_ids;
        tmp["industry"] = creative_map[creative_id].industries;
        for (int i = 1; i <= AGE_NUM; i++) {
            tmp["age_stat"+to_string(i)] = vector<int>(1, creative_map[creative_id].age_stat[i]);
        }
        for (int i = 1; i <= AGE_NUM; i++) {
            tmp["age_ratio"+to_string(i)] = vector<int>(1, creative_map[creative_id].get_age_ratio(i, time-1));
        }
        for (int i = 1; i <= GENDER_NUM; i++) {
            tmp["gender_stat"+to_string(i)] = vector<int>(1, creative_map[creative_id].gender_stat[i]);
        }
        for (int i = 1; i <= GENDER_NUM; i++) {
            tmp["gender_ratio"+to_string(i)] = vector<int>(1, creative_map[creative_id].get_gender_ratio(i, time-1));
        }
        tmp["pv"] = vector<int>(1, creative_map[creative_id].time_pv[time-1]);
        tmp["rcid"] = user_map[user_id].get_creative_ids();
        tmp["raid"] = user_map[user_id].get_ad_ids();
        tmp["rpid"] = user_map[user_id].get_product_ids();
        tmp["rpc"] = user_map[user_id].get_product_categories();
        tmp["radid"] = user_map[user_id].get_advertiser_ids();
        tmp["ri"] = user_map[user_id].get_industries();
        record.pb(tmp);
        count++;
        if (count % batch_size==0) {
            filetotfrecord(count/batch_size, testing);
            indexx = count/batch_size;
            record.clear();
        }
    }
    if (testing) {
        return;
    }
    filetotfrecord(indexx+1, testing);

    cout<<"finish fullfill record, with number: "<<record.size()<<endl;
}

void gen_testing_data() {
    cout<<"begin to fullfill record"<<endl;
    record.clear();
    user_map.clear();
    user_map_cache.clear();
    ifstream ad("/Users/huangqingwei/Documents/C++ workspace/codeforces/test/adz.csv");
    ifstream click_log("/Users/huangqingwei/Documents/C++ workspace/codeforces/test/click_log.csv");
    string line;
    //read_ad_map
    cout<<"begin to read ad_map"<<endl;
    getline(ad, line);
    int creative_id, ad_id, product_id, product_category, advertiser_id, industry;
    while(getline(ad, line)) {
        sscanf(line.c_str(), "%d,%d,%d,%d,%d,%d", &creative_id, &ad_id, &product_id, &product_category, &advertiser_id, &industry);
        if (creative_map.find(creative_id) == creative_map.end()) {
            creative_map[creative_id] = creative_profile();
            creative_map[creative_id].add_ad_ids(ad_id);
            creative_map[creative_id].add_product_id(product_id);
            creative_map[creative_id].add_product_category(product_category);
            creative_map[creative_id].add_advertiser_id(advertiser_id);
            creative_map[creative_id].add_industry(industry);
            for (int i = 0; i < AGE_NUM; i++) {
                creative_map[creative_id].age_stat[i] = 60;
            }
            for (int i = 0; i < GENDER_NUM; i++) {
                creative_map[creative_id].gender_stat[i] = 60;
            }
        }
    }
    //calculate for stat feature
    cout<<"begin to calculate rcid"<<endl;
    getline(click_log, line);
    int time, user_id, click_times;
    int count = 0;
    while(getline(click_log, line)) {
        sscanf(line.c_str(), "%d,%d,%d,%d", &time, &user_id, &creative_id, &click_times);
        if (creative_map.find(creative_id) == creative_map.end()) {
            creative_map[creative_id] = creative_profile();
        }
//        creative_map[creative_id].pv += 1;
        if (user_map.find(user_id) == user_map.end()) {
            user_map.insert({user_id, {0, 0, tmpv, tmpv, tmpv, tmpv, tmpv, tmpv, tmps, tmps, tmps, tmps, tmps, tmps}});
        }
        log_list[count] = {time, user_id, creative_id, click_times};

        count++;
    }
    cout<<"number of log: "<<count<<endl;
    cout<<"begin to generate record vector"<<endl;
    int tot = count;
    sort(log_list, log_list+tot, [](act_log &a, act_log &b){
        if(a.time == b.time)
            return a.creative_id < b.creative_id;
        return a.time < b.time;
    });
    count = 0;
    int current_time = 1;
    for (int h = 0; h < tot; h++) {
        time = log_list[h].time;
        user_id = log_list[h].user_id;
        creative_id = log_list[h].creative_id;
        click_times = log_list[h].click_times;
        if (current_time != time) {
            for(auto v : user_map_cache) {
                user_map[v.first].absorb(v.second);
            }
            user_map_cache.clear();
            current_time = time;
        }

        if (train_creative_id.find(creative_id) != train_creative_id.end()) {
            user_map_cache[user_id].add_creative_id(creative_id);
        }
        if (train_ad_id.find(creative_map[creative_id].ad_ids[0]) != train_ad_id.end()) {
            user_map_cache[user_id].add_ad_id(creative_map[creative_id].ad_ids[0]);
        }
        if (train_product_id.find(creative_map[creative_id].product_ids[0]) != train_product_id.end()) {
            user_map_cache[user_id].add_product_id(creative_map[creative_id].product_ids[0]);
        }
        if (train_product_category.find(creative_map[creative_id].product_categories[0]) != train_product_category.end()) {
            user_map_cache[user_id].add_product_category(creative_map[creative_id].product_categories[0]);
        }
        if (train_advertiser_id.find(creative_map[creative_id].advertiser_ids[0]) != train_advertiser_id.end()) {
            user_map_cache[user_id].add_advertiser_id(creative_map[creative_id].advertiser_ids[0]);
        }
        if (train_industry.find(creative_map[creative_id].industries[0]) != train_industry.end()) {
            user_map_cache[user_id].add_industry(creative_map[creative_id].industries[0]);
        }

        tmp.clear();
        tmp["time"] = vector<int>(1, time);
        tmp["user_id"] = vector<int>(1, user_id);
        tmp["creative_id"] = vector<int>(1, creative_id);
        tmp["age"] = vector<int>(1, user_map[user_id].age);
        tmp["gender"] = vector<int>(1, user_map[user_id].gender);
        tmp["ad_id"] = creative_map[creative_id].ad_ids;
        tmp["product_id"] = creative_map[creative_id].product_ids;
        tmp["product_category"] = creative_map[creative_id].product_categories;
        tmp["advertiser_id"] = creative_map[creative_id].advertiser_ids;
        tmp["industry"] = creative_map[creative_id].industries;
        for (int i = 1; i <= AGE_NUM; i++) {
            tmp["age_stat"+to_string(i)] = vector<int>(1, creative_map[creative_id].age_stat[i]);
        }
        for (int i = 1; i <= GENDER_NUM; i++) {
            tmp["gender_stat"+to_string(i)] = vector<int>(1, creative_map[creative_id].gender_stat[i]);
        }
        for (int i = 1; i <= AGE_NUM; i++) {
            tmp["age_ratio"+to_string(i)] = vector<int>(1, creative_map[creative_id].get_age_ratio(i, time-1));
        }
        for (int i = 1; i <= GENDER_NUM; i++) {
            tmp["gender_ratio"+to_string(i)] = vector<int>(1, creative_map[creative_id].get_gender_ratio(i, time-1));
        }
        tmp["pv"] =vector<int>(1, creative_map[creative_id].time_pv[time-1]);
        tmp["rcid"] = user_map[user_id].get_creative_ids();
        tmp["raid"] = user_map[user_id].get_ad_ids();
        tmp["rpid"] = user_map[user_id].get_product_ids();
        tmp["rpc"] = user_map[user_id].get_product_categories();
        tmp["radid"] = user_map[user_id].get_advertiser_ids();
        tmp["ri"] = user_map[user_id].get_industries();
        record.pb(tmp);
        count++;
        if (count % batch_size==0) {
            filetotfrecord(count/batch_size, true);
            indexx = count/batch_size;
            record.clear();
        }
    }

    filetotfrecord(indexx+1, true);

    cout<<"finish fullfill record, with number: "<<record.size()<<endl;
}


bool replace(std::string& str, const std::string& from, const std::string& to) {
    while(true) {
        size_t start_pos = str.find(from);
        if(start_pos == std::string::npos)
            return false;
        string a = str.substr(0, start_pos);
        if (start_pos + 2 == str.length()) {
            str = a + "0";
        } else {
            str = a + "0" + str.substr(start_pos+2, str.length() - start_pos - 2);
        }
    }
}

void removeNaN(){
    ifstream ad;
    ofstream adz;
//    ad.open("/Users/huangqingwei/Documents/C++ workspace/codeforces/train_preliminary/ad.csv");
//    adz.open("/Users/huangqingwei/Documents/C++ workspace/codeforces/train_preliminary/adz.csv");
    ad.open("/Users/huangqingwei/Documents/C++ workspace/codeforces/test/ad.csv");
    adz.open("/Users/huangqingwei/Documents/C++ workspace/codeforces/test/adz.csv");
    string line;
    while(getline(ad, line))
    {
        replace(line, "\\N", "0");
        adz << line << '\n';
    }
    ad.close();
    adz.close();
}

void gen_second_phase(bool testing) {
    if(user_map.empty())
        return;
    cout<<"Begin to write second_phase"<<endl;
    string file_path = "/Users/huangqingwei/Documents/C++ workspace/codeforces/train_preliminary/training_data_v2/train_serialize.csv";
    if(testing)
        file_path = "/Users/huangqingwei/Documents/C++ workspace/codeforces/test/testing_data_v2/test_serialize.csv";
    ofstream output(file_path.c_str());
    output<<"user_id,age,gender,creative_ids,ad_ids,product_ids,product_categories,advertiser_ids,industries\n";
    for (auto v : user_map) {
        output<<v.first<<',';
        output<<v.second.age<<','<<v.second.gender<<',';
        for(int i = 0; i < v.second.creative_ids.size(); i++) {
            output<<v.second.creative_ids[i];
            if(i!=v.second.creative_ids.size()-1){
                output<<':';
            }
        }
        output<<',';
        for(int i = 0; i < v.second.ad_ids.size(); i++) {
            output<<v.second.ad_ids[i];
            if(i!=v.second.ad_ids.size()-1){
                output<<':';
            }
        }
        output<<',';
        for(int i = 0; i < v.second.product_ids.size(); i++) {
            output<<v.second.product_ids[i];
            if(i!=v.second.product_ids.size()-1){
                output<<':';
            }
        }
        output<<',';
        for(int i = 0; i < v.second.product_categories.size(); i++) {
            output<<v.second.product_categories[i];
            if(i!=v.second.product_categories.size()-1){
                output<<':';
            }
        }
        output<<',';
        for(int i = 0; i < v.second.advertiser_ids.size(); i++) {
            output<<v.second.advertiser_ids[i];
            if(i!=v.second.advertiser_ids.size()-1){
                output<<':';
            }
        }
        output<<',';
        for(int i = 0; i < v.second.industries.size(); i++) {
            output<<v.second.industries[i];
            if(i!=v.second.industries.size()-1){
                output<<':';
            }
        }
        output<<'\n';
    }
    output.close();
    cout<<"finish to generate second phase"<<endl;
}


void filetotfrecord(int index, bool testing) {
    if(record.empty())
        return;

    cout<<"Begin to write tf example "<<index<<endl;

    string file_path = "/Users/huangqingwei/Documents/C++ workspace/codeforces/train_preliminary/training_data/train_serialize_"+to_string(index)+".csv";
    if(testing)
        file_path = "/Users/huangqingwei/Documents/C++ workspace/codeforces/test/testing_data/test_serialize_"+to_string(index)+".csv";
    ofstream output(file_path.c_str());

    //left feature key
    int cnt = 0;
    for (auto vv : record[0]) {
        output<<vv.first;
        output<<(cnt==record[0].size()-1?'\n':',');
        cnt++;
    }
    for (auto v : record) {

        //left feature
        cnt = 0;
        for (auto vv : v) {
            for(int i = 0; i < vv.second.size(); i++) {
                output<<vv.second[i];
                if(i!=vv.second.size()-1){
                    output<<':';
                }
            }
            output<<(cnt==record[0].size()-1?'\n':',');
            cnt++;
        }

    }
    output.close();
    cout<<"finish to generate tf example "<<index<<endl;
}


map<int, map<int,int>> user_age;
map<int, map<int,int>> user_gender;
int get_most(map<int, int> profile) {
    int result = 1;
    int maxx = 0;
    for (auto v : profile) {
        result = (maxx<v.second ? v.first : result);
        maxx = max(maxx, v.second);
    }
    return result;
}

void gen_result() {
    string line;
    int uid;
    int age;
    int gender;
    for(int i = 1; i < 136; i++)
        cout<<"*";
    cout<<endl;
    for(int i = 1; i < 136; i++) {
        cout<<"*"<<std::flush;
        ifstream input("/Users/huangqingwei/Documents/comp/tencentAD/predict_gender/result_" + to_string(i) + ".txt");
        getline(input, line);
        while(getline(input, line)) {
            sscanf(line.c_str(), "%d,%d,%d", &uid, &age, &gender);
//            user_age[uid][age]++;
            user_gender[uid][gender]++;
        }

        ifstream input_age("/Users/huangqingwei/Documents/comp/tencentAD/predict_age/result_" + to_string(i) + ".txt");
        getline(input_age, line);
        while(getline(input_age, line)) {
            sscanf(line.c_str(), "%d,%d,%d", &uid, &age, &gender);
            user_age[uid][age]++;
        }
    }
    ofstream output("/Users/huangqingwei/Documents/comp/tencentAD/submission.csv");
    output<<"user_id,predicted_age,predicted_gender"<<'\n';
    for (auto v : user_age) {
        output<<v.first<<','<<get_most(user_age[v.first])<<','<<get_most(user_gender[v.first])<<'\n';
    }
    output.close();
}

map<int, map<int,int>> user_age_back;
map<int, map<int,int>> user_gender_back;

void gen_result_v2() {
    string line;
    int uid;
    int age;
    int gender;
    int cid;
    for(int i = 1; i < 136; i++)
        cout<<"*";
    cout<<endl;
    for(int i = 1; i < 136; i++) {
        cout<<"*"<<std::flush;
        ifstream input("/Users/huangqingwei/Documents/comp/tencentAD/predict/result_" + to_string(i) + ".txt");
        getline(input, line);
        while(getline(input, line)) {
            sscanf(line.c_str(), "%d,%d,%d,%d", &uid, &age, &gender, &cid);
            if (train_creative_id.find(cid) != train_creative_id.end()) {
                user_gender[uid][gender]++;
                user_age[uid][age]++;
            }
            else {
                user_gender_back[uid][gender]++;
                user_age_back[uid][gender]++;
            }
        }

//        ifstream input_age("/Users/huangqingwei/Documents/comp/tencentAD/predict_age/result_" + to_string(i) + ".txt");
//        getline(input_age, line);
//        while(getline(input_age, line)) {
//            sscanf(line.c_str(), "%d,%d,%d,%d", &uid, &age, &gender, &cid);
//            if (train_creative_id.find(cid) != train_creative_id.end())
//                user_age[uid][age]++;
//            else
//                user_age_back[uid][gender]++;
//        }
    }
    ofstream output("/Users/huangqingwei/Documents/comp/tencentAD/submission.csv");
    output<<"user_id,predicted_age,predicted_gender"<<'\n';
    for (auto v : user_age) {
        output<<v.first<<','<<get_most(user_age[v.first])<<','<<get_most(user_gender[v.first])<<'\n';
    }
    for (auto v : user_age_back) {
        if (user_age.find(v.first) == user_age.end()) {
            output<<v.first<<','<<get_most(user_age_back[v.first])<<','<<get_most(user_gender_back[v.first])<<'\n';
        }
    }
    output.close();
}

set<int> creative_ids[2];
set<int> cid_log[2];
void check() {
    ifstream test("/Users/huangqingwei/Documents/C++ workspace/codeforces/test/adz.csv");
    ifstream test_log("/Users/huangqingwei/Documents/C++ workspace/codeforces/test/click_log.csv");
    ifstream train("/Users/huangqingwei/Documents/C++ workspace/codeforces/train_preliminary/adz.csv");
    ifstream train_log("/Users/huangqingwei/Documents/C++ workspace/codeforces/train_preliminary/click_log.csv");

    string line;
    getline(test, line);
    getline(test_log, line);
    getline(train, line);
    getline(train_log, line);

    int creative_id, ad_id, product_id, product_category, advertiser_id, industry;
    while(getline(test, line)) {
        sscanf(line.c_str(), "%d,%d,%d,%d,%d,%d", &creative_id, &ad_id, &product_id, &product_category, &advertiser_id, &industry);
        creative_ids[1].insert(creative_id);
    }
    int time, user_id, click_times;
    while(getline(test_log, line)) {
        sscanf(line.c_str(), "%d,%d,%d,%d", &time, &user_id, &creative_id, &click_times);
        cid_log[1].insert(creative_id);
    }
    int result = 0;
    for(auto v : cid_log[1]) {
        if (creative_ids[1].find(v) == creative_ids[1].end())
            result++;
    }
    cout<<result<<endl;
}

int main() {
    ios::sync_with_stdio(false);cout.setf(ios::fixed);cout.precision(20);
#ifdef LOCAL
//    freopen("/Users/huangqingwei/Documents/C++ workspace/codeforces/input.txt", "r", stdin);
//    freopen("/Users/huangqingwei/Documents/C++ workspace/codeforces/output.txt", "w", stdout);
#endif
    /*feature clean*/
//    removeNaN();
    /*train data*/
//    gen_training_data(false/*testing*/);
    /*test data*/
    gen_training_data(true/*testing*/);
    gen_testing_data();
    /*result*/
//    gen_result();
    /*resultv2*/
//    gen_training_data(true/*testing*/);
//    gen_result_v2();

#ifdef LOCAL
    cerr << "Time elapsed: " << 1.0 * clock() / CLOCKS_PER_SEC << " s.\n";
#endif
    return 0;
}















