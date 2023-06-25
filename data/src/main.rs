use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Write},
};

const N: usize = 100;

fn main() -> std::io::Result<()> {
    let mut data = Vec::new();

    for i in 0..N {
        let file_name = format!("../tools/in/{:04}.txt", i);
        // println!("processing :{}", file_name);

        let mut f = File::open(file_name)?;

        let mut contents = String::new();
        f.read_to_string(&mut contents)?;

        let mut bt = BTreeMap::new();
        bt.insert("seed", format!("{}", i));
        bt.insert("input", contents);
        data.push(bt);
    }

    let json = serde_json::to_string_pretty(&data)?;
    let mut output = File::create("input.json")?;
    output.write_all(json.as_bytes())?;
    println!("finished!");
    Ok(())
}
