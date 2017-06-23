import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import svm

class_of_worker = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name='class_of_worker', hash_bucket_size=1000)
detailed_industry_recode = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name='detailed_industry_recode', hash_bucket_size=1000)
detailed_occupation_recode = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name='detailed_occupation_recode', hash_bucket_size=1000)
education = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name='education', hash_bucket_size=1000)
enroll_in_edu_inst_last_wk = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name='enroll_in_edu_inst_last_wk', hash_bucket_size=1000)
marital_stat = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name='marital_stat', hash_bucket_size=1000)
major_industry_code = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name='major_industry_code', hash_bucket_size=1000)
major_occupation_code = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name='major_occupation_code', hash_bucket_size=1000)
race = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name='race', hash_bucket_size=1000)
hispanic_origin = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name='hispanic_origin', hash_bucket_size=1000)
sex = tf.contrib.layers.sparse_column_with_keys(
        column_name='sex', keys=['Female', 'Male'])
member_of_labor_union = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name='member_of_labor_union', hash_bucket_size=1000)
reason_for_unemployment = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name='reason_for_unemployment', hash_bucket_size=1000)
full_or_part_time_employment_stat = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name='full_or_part_time_employment_stat', hash_bucket_size=1000)
tax_filer_stat = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name='tax_filer_stat', hash_bucket_size=1000)
region_of_previous_residence = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name='region_of_previous_residence', hash_bucket_size=1000)
state_of_previous_residence = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name='state_of_previous_residence', hash_bucket_size=1000)
detailed_household_and_family_stat = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name='detailed_household_and_family_stat', hash_bucket_size=1000)
detailed_household_summary_in_household = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name='detailed_household_summary_in_household', hash_bucket_size=1000)
migration_code_change_in_msa = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name='migration_code_change_in_msa', hash_bucket_size=1000)
migration_code_change_in_msa = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name='migration_code_change_in_msa', hash_bucket_size=1000)
migration_code_change_in_reg = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name='migration_code_change_in_reg', hash_bucket_size=1000)
migration_code_move_within_reg = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name='migration_code_move_within_reg', hash_bucket_size=1000)
live_in_this_house_1year_ago = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name='live_in_this_house_1year_ago', hash_bucket_size=1000)
migration_prev_res_in_sunbelt = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name='migration_prev_res_in_sunbelt', hash_bucket_size=1000)
family_members_under18 = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name='family_members_under18', hash_bucket_size=1000)
country_of_birth_father = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name='country_of_birth_father', hash_bucket_size=1000)
country_of_birth_mother = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name='country_of_birth_mother', hash_bucket_size=1000)
country_of_birth_self = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name='country_of_birth_self', hash_bucket_size=1000)
citizenship = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name='citizenship', hash_bucket_size=1000)
own_business_or_self_employed = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name='own_business_or_self_employed', hash_bucket_size=1000)
fill_inc_questionnaire_for_veteran_admin = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name='fill_inc_questionnaire_for_veteran_admin', hash_bucket_size=1000)
veterans_benefits = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name='veterans_benefits', hash_bucket_size=1000)
year = tf.contrib.layers.sparse_column_with_keys(
        column_name='year', keys=['94', '95'])

age = tf.contrib.layers.real_valued_column('age')
age_buckets = tf.contrib.layers.bucketized_column(
        age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
wage_per_hour = tf.contrib.layers.real_valued_column('wage_per_hour')
capital_gains = tf.contrib.layers.real_valued_column('capital_gains')
capital_losses = tf.contrib.layers.real_valued_column('capital_losses')
dividends_from_stocks = tf.contrib.layers.real_valued_column('dividends_from_stocks')
instance_weight = tf.contrib.layers.real_valued_column('instance_weight')
weeks_worked_in_year = tf.contrib.layers.real_valued_column('weeks_worked_in_year')
num_persons_worked_for_employer = tf.contrib.layers.real_valued_column('num_persons_worked_for_employer')


COLUMNS = [
    'age', 'class_of_worker', 'detailed_industry_recode',
    'detailed_occupation_recode', 'education', 'wage_per_hour',
    'enroll_in_edu_inst_last_wk', 'marital_stat', 'major_industry_code',
    'major_occupation_code', 'race', 'hispanic_origin', 'sex',
    'member_of_labor_union', 'reason_for_unemployment',
    'full_or_part_time_employment_stat', 'capital_gains',
    'capital_losses', 'dividends_from_stocks', 'tax_filer_stat',
    'region_of_previous_residence', 'state_of_previous_residence',
    'detailed_household_and_family_stat', 'detailed_household_summary_in_household',
    'instance_weight', 'migration_code_change_in_msa', 'migration_code_change_in_reg',
    'migration_code_move_within_reg', 'live_in_this_house_1year_ago',
    'migration_prev_res_in_sunbelt', 'num_persons_worked_for_employer',
    'family_members_under18', 'country_of_birth_father',
    'country_of_birth_mother', 'country_of_birth_self',
    'citizenship', 'own_business_or_self_employed',
    'fill_inc_questionnaire_for_veteran_admin', 'veterans_benefits',
    'weeks_worked_in_year', 'year', 'label'
]

FEATURE_COLUMNS = [
    age, age_buckets, class_of_worker, detailed_industry_recode,
    detailed_occupation_recode, education, wage_per_hour,
    enroll_in_edu_inst_last_wk, marital_stat, major_industry_code,
    major_occupation_code, race, hispanic_origin, sex, member_of_labor_union,
    reason_for_unemployment, full_or_part_time_employment_stat,
    capital_gains, capital_losses, dividends_from_stocks, tax_filer_stat,
    region_of_previous_residence, state_of_previous_residence,
    detailed_household_and_family_stat, detailed_household_summary_in_household,
    instance_weight, migration_code_change_in_msa,
    migration_code_change_in_reg, migration_code_move_within_reg,
    live_in_this_house_1year_ago, migration_prev_res_in_sunbelt,
    num_persons_worked_for_employer, family_members_under18,
    country_of_birth_father, country_of_birth_mother,
    country_of_birth_self, citizenship, own_business_or_self_employed,
    fill_inc_questionnaire_for_veteran_admin, veterans_benefits,
    weeks_worked_in_year, year
]
LABEL_COLUMN = 'label'

CONTINUOUS_COLUMNS = [
    'age', 'wage_per_hour', 'capital_gains', 'capital_losses',
    'dividends_from_stocks', 'instance_weight', 'weeks_worked_in_year',
    'num_persons_worked_for_employer'
]
CATEGORICAL_COLUMNS = [
    'class_of_worker', 'detailed_industry_recode',
    'detailed_occupation_recode', 'education', 'enroll_in_edu_inst_last_wk',
    'marital_stat', 'major_industry_code', 'major_occupation_code', 'race',
    'hispanic_origin', 'sex', 'member_of_labor_union',
    'reason_for_unemployment', 'full_or_part_time_employment_stat',
    'tax_filer_stat', 'region_of_previous_residence',
    'state_of_previous_residence', 'detailed_household_and_family_stat',
    'detailed_household_summary_in_household', 'migration_code_change_in_msa',
    'migration_code_change_in_reg', 'migration_code_move_within_reg',
    'live_in_this_house_1year_ago', 'migration_prev_res_in_sunbelt',
    'family_members_under18', 'country_of_birth_father',
    'country_of_birth_mother', 'country_of_birth_self', 'citizenship',
    'own_business_or_self_employed', 'fill_inc_questionnaire_for_veteran_admin',
    'veterans_benefits', 'year'
]

print len(COLUMNS)
print len(FEATURE_COLUMNS)
print len(CATEGORICAL_COLUMNS)
print len(CONTINUOUS_COLUMNS)

TRAIN_FILE = './data/census-income.data'
TEST_FILE = './data/census-income.test'


def dense_to_sparse(dense_tensor):
    indices = tf.to_int64(tf.transpose([tf.range(tf.shape(dense_tensor)[0]), 
        tf.zeros_like(dense_tensor, dtype=tf.int32)]))
    values = dense_tensor
    shape = tf.to_int64([tf.shape(dense_tensor)[0], tf.constant(1)])

    return tf.SparseTensor(indices=indices, values=values, dense_shape=shape)

def input_fn(batch_size, filenames):
    examples_op = tf.contrib.learn.read_batch_examples(
            filenames,
            batch_size=batch_size,
            reader=tf.TextLineReader,
            num_epochs=1,
            parse_fn=lambda x: tf.decode_csv(x,[tf.constant([''], dtype=tf.string)] * len(COLUMNS))
    )

    examples_dict = {}
    for i, header in enumerate(COLUMNS):
        examples_dict[header] = examples_op[:, i]

    feature_cols = {k: tf.string_to_number(examples_dict[k], out_type=tf.float32) for k in CONTINUOUS_COLUMNS}

    feature_cols.update({k: dense_to_sparse(examples_dict[k]) for k in CATEGORICAL_COLUMNS})

    label = examples_dict[LABEL_COLUMN]
    label = tf.case({
        tf.equal(label, tf.constant('- 50000.')): lambda: tf.constant(0),
        tf.equal(label, tf.constant('50000+.')): lambda: tf.constant(1),
    }, lambda: tf.constant(-1), exclusive=True)

    return feature_cols, label

def train_input_fn():
    batch_size = 128
    return input_fn(batch_size, [TRAIN_FILE])

def eval_input_fn():
    batch_size = 5000
    return input_fn(batch_size, [TEST_FILE])

model_dir = './model'
model = tf.contrib.learn.LinearClassifier(
    feature_columns=FEATURE_COLUMNS,
    model_dir=model_dir
)
model.fit(input_fn=train_input_fn, steps=200)
results = model.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))

