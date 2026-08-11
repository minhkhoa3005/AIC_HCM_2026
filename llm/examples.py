"""Day 1 sample queries for task classification and schema alignment."""

DAY1_QUERY_EXAMPLES = [
    {
        "input": "Tim canh nguoi dan ong mac ao do mo cua xe mau trang.",
        "expected_task_type": "TEXTUAL_KIS",
        "notes": "KIS voi nguoi, mau ao, xe va hanh dong mo cua.",
    },
    {
        "input": "Tim khoanh khac mot em be cam bong bay mau vang trong cong vien.",
        "expected_task_type": "TEXTUAL_KIS",
        "notes": "Truy van can doi tuong chinh va ngu canh cong vien.",
    },
    {
        "input": "Tim canh xe may chay qua nga tu vao buoi toi.",
        "expected_task_type": "TEXTUAL_KIS",
        "notes": "Can vat the, hanh dong, dia diem va thoi diem.",
    },
    {
        "input": "Tim nguoi phu nu doi mu bao hiem dang dung canh xe dap.",
        "expected_task_type": "TEXTUAL_KIS",
        "notes": "Mo ta visual truc tiep, khong hoi dap.",
    },
    {
        "input": "Tim canh nhieu nguoi xep hang truoc quay ban ve.",
        "expected_task_type": "TEXTUAL_KIS",
        "notes": "Can objects va scene trong khung hinh.",
    },
    {
        "input": "Tim hinh anh con cho nam duoi gam ban trong quan ca phe.",
        "expected_task_type": "TEXTUAL_KIS",
        "notes": "Can quan he vi tri giua doi tuong va scene.",
    },
    {
        "input": "Tim canh nguoi mac ao mua xanh di bo tren duong.",
        "expected_task_type": "TEXTUAL_KIS",
        "notes": "KIS dua tren trang phuc, hanh dong, duong pho.",
    },
    {
        "input": "Tim chiec xe buyt mau vang dung o tram xe.",
        "expected_task_type": "TEXTUAL_KIS",
        "notes": "Object-centric search.",
    },
    {
        "input": "Tim canh mot nguoi dang chup anh mon an tren ban.",
        "expected_task_type": "TEXTUAL_KIS",
        "notes": "Action attached to person with food/table objects.",
    },
    {
        "input": "Tim nguoi cong nhan mac ao phan quang tren cong truong.",
        "expected_task_type": "TEXTUAL_KIS",
        "notes": "Scene cong truong va thuoc tinh ao.",
    },
    {
        "input": "Tim canh tau hoa di qua cau.",
        "expected_task_type": "TEXTUAL_KIS",
        "notes": "Vehicle, motion action, bridge scene.",
    },
    {
        "input": "Tim canh nguoi ban hang dua tui do cho khach.",
        "expected_task_type": "TEXTUAL_KIS",
        "notes": "Quan he giua nguoi ban, khach va tui do.",
    },
    {
        "input": "Tim hinh anh ban hoc co may tinh xach tay va sach.",
        "expected_task_type": "TEXTUAL_KIS",
        "notes": "Static scene search.",
    },
    {
        "input": "Tim canh nguoi dang cho xang cho xe may.",
        "expected_task_type": "TEXTUAL_KIS",
        "notes": "Action target la xe may.",
    },
    {
        "input": "Tim canh mot nguoi cam o di qua vach sang duong.",
        "expected_task_type": "TEXTUAL_KIS",
        "notes": "Person action with umbrella and crosswalk.",
    },
    {
        "input": "Tim khung hinh co bien hieu mau do phia tren cua hang.",
        "expected_task_type": "TEXTUAL_KIS",
        "notes": "Object va vi tri trong scene.",
    },
    {
        "input": "Tim canh nhan vien nha hang dat dia thuc an len ban.",
        "expected_task_type": "TEXTUAL_KIS",
        "notes": "Entity action with target plate/table.",
    },
    {
        "input": "Tim canh nguoi choi bong ro nem bong vao ro.",
        "expected_task_type": "TEXTUAL_KIS",
        "notes": "Sport action query.",
    },
    {
        "input": "Tim chiec thuyen nho tren song gan cau go.",
        "expected_task_type": "TEXTUAL_KIS",
        "notes": "Outdoor scene with vehicle and landmark.",
    },
    {
        "input": "Tim canh nguoi mac ao trang ngoi tren ghe ben cua so.",
        "expected_task_type": "TEXTUAL_KIS",
        "notes": "Person attributes and spatial scene.",
    },
    {
        "input": "Trong canh nguoi phu nu dung truoc xe buyt, bien so xe la gi?",
        "expected_task_type": "QA",
        "notes": "Can retrieve visual evidence then answer OCR/visual detail.",
    },
    {
        "input": "Nguoi dan ong dang cam vat gi khi buoc vao cua hang?",
        "expected_task_type": "QA",
        "notes": "Question asks for object held by person.",
    },
    {
        "input": "Mau cua chiec xe dang do truoc cong la mau gi?",
        "expected_task_type": "QA",
        "notes": "Visual attribute question.",
    },
    {
        "input": "Co bao nhieu nguoi dang ngoi quanh ban an?",
        "expected_task_type": "QA",
        "notes": "Counting question over retrieved candidate.",
    },
    {
        "input": "Dong chu tren bien hieu cua quan la gi?",
        "expected_task_type": "QA",
        "notes": "OCR-oriented QA.",
    },
    {
        "input": "Nguoi mac ao xanh dang dung ben trai hay ben phai khung hinh?",
        "expected_task_type": "QA",
        "notes": "Spatial answer required.",
    },
    {
        "input": "Con vat nam tren ghe la loai gi?",
        "expected_task_type": "QA",
        "notes": "Object category answer.",
    },
    {
        "input": "Nguoi giao hang dang di bang phuong tien nao?",
        "expected_task_type": "QA",
        "notes": "Question asks vehicle type.",
    },
    {
        "input": "Cua hang trong video ban loai san pham nao?",
        "expected_task_type": "QA",
        "notes": "Scene understanding plus possible OCR.",
    },
    {
        "input": "Thoi tiet trong canh nguoi cam o co ve nhu the nao?",
        "expected_task_type": "QA",
        "notes": "Answer inferred from visual context.",
    },
    {
        "input": "Nguoi dau bep dang cat nguyen lieu gi?",
        "expected_task_type": "QA",
        "notes": "Entity action target answer.",
    },
    {
        "input": "Man hinh may tinh dang hien thi noi dung gi?",
        "expected_task_type": "QA",
        "notes": "Screen/OCR detail answer.",
    },
    {
        "input": "Chiec xe tai co logo hay chu nao tren than xe?",
        "expected_task_type": "QA",
        "notes": "Visual/OCR detail question.",
    },
    {
        "input": "Nguoi dang dung tren san khau cam micro bang tay nao?",
        "expected_task_type": "QA",
        "notes": "Fine-grained visual question.",
    },
    {
        "input": "Trong canh phong hop, tren tuong co dong ho khong?",
        "expected_task_type": "QA",
        "notes": "Yes/no visual question.",
    },
    {
        "input": "Tim chuoi su kien nguoi mo cua xe, buoc ra ngoai, roi di vao toa nha.",
        "expected_task_type": "TRAKE",
        "notes": "Three ordered events.",
    },
    {
        "input": "Tim doan nguoi lay dien thoai ra, chup anh, sau do cat dien thoai vao tui.",
        "expected_task_type": "TRAKE",
        "notes": "Temporal sequence with same entity.",
    },
    {
        "input": "Tim canh xe dung lai, nguoi buoc xuong, roi lay hang tu cop xe.",
        "expected_task_type": "TRAKE",
        "notes": "Ordered vehicle/person/object events.",
    },
    {
        "input": "Tim chuoi nguoi di vao bep, mo tu lanh, lay chai nuoc.",
        "expected_task_type": "TRAKE",
        "notes": "Indoor action sequence.",
    },
    {
        "input": "Tim doan nguoi dat vali xuong, mo khoa, roi lay quan ao ra.",
        "expected_task_type": "TRAKE",
        "notes": "Manipulation sequence.",
    },
    {
        "input": "Tim canh cau thu nhan bong, chay den ro, va nem bong.",
        "expected_task_type": "TRAKE",
        "notes": "Sports temporal order.",
    },
    {
        "input": "Tim doan nguoi di qua duong, len via he, roi vao cua hang.",
        "expected_task_type": "TRAKE",
        "notes": "Location transition sequence.",
    },
    {
        "input": "Tim chuoi em be nhat bong, chay ve phia cau truot, roi tha bong xuong.",
        "expected_task_type": "TRAKE",
        "notes": "Multiple ordered child actions.",
    },
    {
        "input": "Tim canh nhan vien nhan goi hang, quet ma, sau do dua cho khach.",
        "expected_task_type": "TRAKE",
        "notes": "Work-flow sequence with package.",
    },
    {
        "input": "Tim doan nguoi bat den, ngoi vao ban, roi mo laptop.",
        "expected_task_type": "TRAKE",
        "notes": "Ordered indoor activity.",
    },
    {
        "input": "Tim chuoi xe buyt toi tram, cua mo ra, hanh khach buoc xuong.",
        "expected_task_type": "TRAKE",
        "notes": "Vehicle arrival and passenger action.",
    },
    {
        "input": "Tim canh nguoi rua rau, thai rau, roi cho vao noi.",
        "expected_task_type": "TRAKE",
        "notes": "Cooking preparation sequence.",
    },
    {
        "input": "Tim doan nguoi dat sach len ke, sap xep lai, va dong cua tu.",
        "expected_task_type": "TRAKE",
        "notes": "Object arrangement sequence.",
    },
    {
        "input": "Tim chuoi cho chay den nguoi chu, nguoi chu cui xuong, roi deo day co.",
        "expected_task_type": "TRAKE",
        "notes": "Interaction sequence between entities.",
    },
    {
        "input": "Tim canh nguoi mo hop banh, lay mot cai banh, sau do dua cho dua tre.",
        "expected_task_type": "TRAKE",
        "notes": "Ordered transfer action.",
    },
]
