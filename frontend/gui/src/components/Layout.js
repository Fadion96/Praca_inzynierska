import React from 'react';
import classnames from 'classnames';
import {
    Button,
    Container,
    Menu,
} from 'semantic-ui-react'

const AppMenu = (props) => {
    const appMenuClass = classnames('app_menu');
    return (
        <Menu inverted icon className={appMenuClass}>
            <Container fluid>
                <Menu.Item header>
                    Aplikacja przetwarzająca obrazy
                </Menu.Item>
                <Menu.Item>
                    <label className="custom-file-upload">
                        <input ref={props.inputImageRef} type='file' accept='image/*' onChange={props.imagesHandler}/>
                        <i className="file image outline icon"/> Załaduj obraz
                    </label>
                </Menu.Item>
                <Menu.Item>
                    <Button
                        inverted
                        basic
                        content='Wykonaj algorytm'
                        onClick={props.handleSend}
                    />
                </Menu.Item>
                <Menu.Item>
                    <Button
                        inverted
                        basic
                        content='Wyczyść'
                        color='red'
                        onClick={props.handleDelete}
                    />
                </Menu.Item>
                <Menu.Item>
                    <label className="custom-file-upload">
                        <input type='file' accept='.py' onChange={props.fileHandler}/>
                        <i className="code icon"/> Załaduj własny blok
                    </label>
                </Menu.Item>
                <Menu.Item>
                    <Button
                        inverted
                        basic
                        content='Zapisz algorytm'
                        onClick={props.saveAlgorithm}
                    />
                </Menu.Item>
                     <Menu.Item>
                    <label className="custom-file-upload">
                        <input type='file' accept='.json' onChange={props.jsonHandler}/>
                        <i className="code icon"/> Załaduj algorytm
                    </label>
                </Menu.Item>
            </Container>
        </Menu>
    );
};

export default AppMenu
